"""Train SHARoadSeg or its native baseline from external official pretrained weights."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from SHARoadSeg.data import RoadDataset
from SHARoadSeg.inference import setup_runtime
from SHARoadSeg.model import build_model
from SHARoadSeg.qmr import material_nll
from SHARoadSeg.training import (EMA, accumulation_divisor, atomic_save, build_mask_targets,
    make_scheduler, optimizer_groups, pure_material_targets, restore_rng, rng_state, seed_all, validate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/sharoadseg.json")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, default=Path(__file__).parent / "weights/pretrained",
                        help="Local official Cityscapes checkpoint directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    setup_runtime()
    recipe = json.loads(args.config.read_text())
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    torch.cuda.set_device(device)
    if recipe["precision"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("SHARoadSeg training requires BF16 support")
    if recipe["whole_probability"] > 0 and recipe["batch_size"] != 1:
        raise ValueError("Variable whole-image geometry requires batch_size=1")
    if recipe["workers"] != 0 and args.resume:
        raise ValueError("Exact epoch-boundary RNG resume requires workers=0")
    if not args.resume and not args.pretrained:
        parser.error("A fresh run requires --pretrained")
    if args.resume:
        state = torch.load(args.out / "last.pt", map_location="cpu", weights_only=True)
        if state["recipe"] != recipe or state.get("finished", False):
            raise ValueError("Resume recipe differs or training has completed")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        state = None
    seed_all(recipe["seed"])
    train = RoadDataset(args.data, "train", training=True, deduplicate=recipe["deduplicate_train"],
                        whole_probability=recipe["whole_probability"])
    val = RoadDataset(args.data, "val")
    print(json.dumps({"train_images": len(train), "validation_images": len(val),
                      "removed_training_duplicates": len(train.excluded)}), flush=True)
    if state and (state["train_ids"] != train.ids or state["val_ids"] != val.ids):
        raise ValueError("Resume dataset IDs differ")
    model = build_model(recipe["variant"], pretrained=args.pretrained if not state else None,
                        seed=recipe["seed"]).float().to(device).requires_grad_(True)
    generator = torch.Generator().manual_seed(recipe["seed"])
    loader = DataLoader(train, batch_size=recipe["batch_size"], shuffle=True,
                        num_workers=recipe["workers"], generator=generator,
                        persistent_workers=recipe["workers"] > 0)
    optimizer = torch.optim.AdamW(optimizer_groups(model, recipe), betas=(.9, .999))
    scheduler = make_scheduler(optimizer, len(loader), recipe)
    ema = EMA(model, recipe["ema_decay"])
    use_scaler = recipe["precision"] == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    dtype = torch.float16 if use_scaler else torch.bfloat16
    best, bad, start, history = None, 0, 1, []
    if state:
        model.load_state_dict(state["raw_model"], strict=True)
        ema.module.load_state_dict(state["ema_model"], strict=True)
        ema.steps = state["ema_steps"]
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        best, bad, start, history = state["best_score"], state["bad_epochs"], state["epoch"] + 1, state["history"]
        if state["best_checkpoint"] is not None:
            atomic_save(args.out / "best.pt", state["best_checkpoint"])
        restore_rng(state["rng"], generator)
        del state
    for epoch in range(start, recipe["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for step, sample in enumerate(loader):
            images, labels = sample["image"].to(device), sample["label"].to(device)
            masks, classes = build_mask_targets(labels)
            with torch.autocast("cuda", dtype=dtype):
                if recipe["variant"] == "baseline":
                    output = model(images, mask_labels=masks, class_labels=classes)
                    loss = output.loss.float()
                else:
                    output, adapted = model(images, labels != 255, masks, classes)
                    auxiliary, _ = material_nll(adapted, pure_material_targets(labels, adapted.base_scores.shape[-2:]))
                    loss = output.loss.float() + auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            accum = recipe["accumulation"]
            divisor = accumulation_divisor(step, len(loader), accum) if recipe["normalize_short_window"] else accum
            scaler.scale(loss / divisor).backward()
            loss_sum += float(loss.detach())
            if (step + 1) % accum == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), recipe["clip_norm"], error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            if step == 0 or (step + 1) % 100 == 0 or step + 1 == len(loader):
                print(f"Epoch {epoch}: {step + 1}/{len(loader)} loss={loss_sum / (step + 1):.5f}", flush=True)
        before_validation = rng_state(generator)
        score = validate(ema.module, val, device, recipe)
        restore_rng(before_validation, generator)
        if not math.isfinite(score):
            raise FloatingPointError("Validation mIoU is undefined")
        if best is None or score > best:
            best, bad = score, 0
            atomic_save(args.out / "best.pt", {"variant": recipe["variant"], "model": ema.module.state_dict(),
                        "epoch": epoch, "best_score": best})
        elif epoch >= recipe["patience_start_epoch"]:
            bad += 1
        finished = bad >= recipe["patience"] or epoch == recipe["epochs"]
        history.append({"epoch": epoch, "loss": loss_sum / len(loader), "val_mIoU_fg": score, "best_score": best})
        atomic_save(args.out / "last.pt", {"recipe": recipe, "variant": recipe["variant"], "epoch": epoch,
                    "raw_model": model.state_dict(), "ema_model": ema.module.state_dict(), "ema_steps": ema.steps,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "best_score": best, "bad_epochs": bad, "rng": rng_state(generator), "history": history,
                    "train_ids": train.ids, "val_ids": val.ids, "finished": finished,
                    "best_checkpoint": torch.load(args.out / "best.pt", map_location="cpu", weights_only=True)})
        (args.out / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
        print(json.dumps(history[-1]), flush=True)
        if finished:
            break


if __name__ == "__main__":
    main()
