"""Optimization, material supervision, EMA, and resumable training state."""
import copy
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import top_quarter_background
from .geometry import letterbox
from .metrics import iou
from .qmr import input_valid_mask, resize_adapter_scores, refined_labels


def build_mask_targets(labels):
    masks, classes = [], []
    for label in labels:
        present = torch.unique(label)
        present = present[(present >= 0) & (present < 4)]
        if not len(present):
            present = label.new_zeros(1)
        masks.append(torch.stack([(label == c).float() for c in present]))
        classes.append(present.long())
    return masks, classes


def pure_material_targets(labels, size):
    valid = (labels >= 0) & (labels < 4)
    safe = labels.masked_fill(~valid, 0)
    hot = F.one_hot(safe, 4).permute(0, 3, 1, 2).float() * valid[:, None]
    histogram = F.interpolate(hot, size=size, mode="area")
    confidence, category = histogram.max(1)
    return (category - 1).masked_fill(~((confidence >= .99) & (category > 0)), -100)


def accumulation_divisor(step, steps, accumulation):
    return min(accumulation, steps - (step // accumulation) * accumulation)


class EMA:
    def __init__(self, module, decay=.9998):
        self.module = copy.deepcopy(module).eval().requires_grad_(False)
        self.decay, self.steps = decay, 0

    @torch.no_grad()
    def update(self, module):
        self.steps += 1
        decay = min(self.decay, (1 + self.steps) / (10 + self.steps))
        source = module.state_dict()
        for key, value in self.module.state_dict().items():
            if value.is_floating_point():
                value.mul_(decay).add_(source[key].detach(), alpha=1 - decay)
            else:
                value.copy_(source[key])


def optimizer_groups(model, recipe):
    native = model if recipe["variant"] == "baseline" else model.semantic
    backbone = list(native.backbone_parameters())
    ids = {id(p) for p in backbone}
    groups = [dict(params=backbone, lr=recipe["backbone_lr"],
                   weight_decay=recipe["native_weight_decay"], name="backbone"),
              dict(params=[p for p in native.parameters() if id(p) not in ids],
                   lr=recipe["native_lr"], weight_decay=recipe["native_weight_decay"], name="native")]
    if recipe["variant"] == "sharoadseg":
        for name, module in (("sha", model.sha), ("qmr", model.adapter)):
            groups.append(dict(params=list(module.parameters()), lr=recipe["module_lr"],
                               weight_decay=recipe["module_weight_decay"], name=name))
    assigned = [id(p) for group in groups for p in group["params"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != {id(p) for p in model.parameters()}:
        raise AssertionError("Optimizer parameter coverage differs from model")
    return groups


def make_scheduler(optimizer, steps, recipe):
    total = math.ceil(steps / recipe["accumulation"]) * recipe["epochs"]
    warmup = recipe["warmup_updates"]
    if recipe["variant"] == "baseline":
        warmup = max(0, min(warmup, total - 1))
    def factor(step):
        if step < warmup:
            return .001 + .999 * step / warmup
        progress = min(1., max(0., (step - warmup) / max(1, total - warmup)))
        return .5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state(generator):
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def atomic_save(path, state):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


@torch.inference_mode()
def validate(model, dataset, device, recipe):
    model.eval()
    total = np.zeros((4, 4), np.int64)
    for index in range(len(dataset)):
        sample = dataset[index]
        image, vh, vw = letterbox(sample["image"][None].to(device), recipe["validation_long_side"], 32)
        valid = input_valid_mask(*image.shape[-2:], vh, vw, device=device)
        with torch.autocast(device_type=device.type, enabled=False):
            if recipe["variant"] == "baseline":
                base = model.predict_logits(image)
                delta = torch.zeros_like(base[:, 1:])
            else:
                _, adapted = model(image, valid)
                base, delta = adapted.base_scores, adapted.delta_scores
            native, refined = resize_adapter_scores(base, delta, image.shape[-2:], (vh, vw), sample["size"])
            pred = native.argmax(1) if recipe["variant"] == "baseline" else refined_labels(native, refined)
        pred = pred[0].cpu().numpy()
        gt = sample["label"].numpy()
        if recipe["validation_top_quarter_background"]:
            pred, gt = top_quarter_background(pred), top_quarter_background(gt)
        keep = (gt >= 0) & (gt < 4)
        total += np.bincount(4 * gt[keep].astype(np.int64) + pred[keep].astype(np.int64), minlength=16).reshape(4, 4)
        if (index + 1) % 200 == 0 or index + 1 == len(dataset):
            print(f"Validation {index + 1}/{len(dataset)}", flush=True)
    return float(np.nanmean(iou(total)[1:]))
