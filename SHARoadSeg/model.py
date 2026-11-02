"""Mask2Former-Swin-T baseline and the complete SHA + QMR architecture."""
from pathlib import Path
import json

import torch
from torch import nn
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

from .sha import ShoulderAwareDecoder
from .qmr import QueryMaterialResidual, normalized_query_embeddings, semantic_scores

CLASS_NAMES = {0: "background", 1: "asphalt", 2: "concrete", 3: "unpaved"}
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs/mask2former.json"


class Mask2FormerBaseline(nn.Module):
    """Native four-class model; state keys retain the ``net.`` prefix."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def backbone_parameters(self):
        return self.net.model.pixel_level_module.encoder.parameters()

    def forward(self, images, valid_mask=None, mask_labels=None, class_labels=None):
        return self.net(pixel_values=images, mask_labels=mask_labels,
                        class_labels=class_labels, output_hidden_states=True,
                        return_dict=True)

    @torch.no_grad()
    def predict_logits(self, images):
        output = self.net(pixel_values=images, return_dict=True)
        return semantic_scores(output.class_queries_logits, output.masks_queries_logits)


class SHARoadSeg(ShoulderAwareDecoder):
    """One native forward with shared SHA and query-level material refinement."""

    def __init__(self, semantic):
        super().__init__(semantic)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(4204)
            self.adapter = QueryMaterialResidual()
        self.adapter.to(next(semantic.parameters()).device)

    def forward(self, images, valid_mask, mask_labels=None, class_labels=None):
        output = super().forward(images, valid_mask, mask_labels, class_labels)
        adapted = self.adapter(normalized_query_embeddings(self.semantic, output),
                               output.class_queries_logits, output.masks_queries_logits)
        return output, adapted


def build_model(variant="sharoadseg", *, pretrained=None, config=DEFAULT_CONFIG, seed=42):
    """Build locally; ``pretrained`` is an external official Cityscapes directory.

    The classifier and loss class weights are rebuilt for four semantic classes.
    Omitting pretrained creates an architecture for strict checkpoint loading.
    """
    if variant not in ("baseline", "sharoadseg"):
        raise ValueError("variant must be baseline or sharoadseg")
    source = Path(pretrained) / "config.json" if pretrained else Path(config)
    cfg = Mask2FormerConfig.from_dict(json.loads(source.read_text(encoding="utf-8")))
    cfg.num_labels = 4
    cfg.id2label = CLASS_NAMES.copy()
    cfg.label2id = {v: k for k, v in CLASS_NAMES.items()}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if pretrained:
            net = Mask2FormerForUniversalSegmentation.from_pretrained(
                str(pretrained), config=cfg, local_files_only=True,
                ignore_mismatched_sizes=True)
        else:
            net = Mask2FormerForUniversalSegmentation(cfg)
    semantic = Mask2FormerBaseline(net)
    return semantic if variant == "baseline" else SHARoadSeg(semantic)


def load_checkpoint(path, variant, device="cpu"):
    """Load a tensor checkpoint or a training checkpoint with a ``model`` field."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "variant" in state and state["variant"] != variant:
        raise ValueError("Checkpoint variant differs from requested model")
    model = build_model(variant)
    model.load_state_dict(state.get("model", state), strict=True)
    return model.float().to(device).eval()
