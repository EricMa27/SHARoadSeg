"""Query score synthesis, material loss, and precision-preserving readout."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


class QueryMaterialResidual(nn.Module):
    """840-to-256 descriptor MLP with bounded residual and sigmoid gate.

    The descriptor contains query256, constant zeros576, and confidence8.
    Native logits and masks are detached; query embeddings remain differentiable.
    """

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(840, 256), nn.GELU(), nn.Dropout(.1))
        self.residual_head = nn.Linear(256, 3)
        self.gate_head = nn.Linear(256, 1)
        for head in (self.residual_head, self.gate_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, query_embeddings, class_logits, mask_logits):
        if query_embeddings.shape[-1] != 256 or class_logits.shape[-1] != 5:
            raise ValueError("Expected query256 and four classes plus no-object")
        with torch.autocast(device_type=query_embeddings.device.type, enabled=False):
            logits = class_logits.detach().float()
            masks = mask_logits.detach().float().sigmoid()
            probability = logits.softmax(-1)
            mass = probability[..., 1:4].sum(-1, keepdim=True)
            conditional = probability[..., 1:4] / mass.clamp_min(1e-12)
            entropy = -(conditional * conditional.clamp_min(1e-12).log()).sum(-1, keepdim=True)
            top2 = conditional.topk(2, dim=-1).values
            confidence = torch.cat((conditional, mass, entropy, top2[..., :1] - top2[..., 1:2],
                                    probability[..., :1], probability[..., 4:5]), -1)
            zeros = query_embeddings.new_zeros((*query_embeddings.shape[:2], 576), dtype=torch.float32)
            hidden = self.body(torch.cat((query_embeddings.float(), zeros, confidence), -1))
            residual = 4. * torch.tanh(self.residual_head(hidden) / 4.)
            gate = self.gate_head(hidden).sigmoid()
            query_delta = mass * ((logits[..., 1:4] + gate * residual).softmax(-1)
                                  - logits[..., 1:4].softmax(-1))
            delta = torch.bmm(query_delta.transpose(1, 2), masks.flatten(2))
            delta = delta.reshape(query_embeddings.shape[0], 3, *mask_logits.shape[-2:])
            base = semantic_scores(logits, mask_logits.detach())
            refined = torch.cat((base[:, :1], base[:, 1:] + delta), 1)
            if not torch.isfinite(refined).all():
                raise FloatingPointError("Non-finite QMR scores")
        return QueryAdapterOutput(base, delta, refined, residual, gate, query_delta)

@dataclass
class QueryAdapterOutput:
    base_scores: torch.Tensor
    delta_scores: torch.Tensor
    refined_scores: torch.Tensor
    residual: torch.Tensor
    gate: torch.Tensor
    query_delta: torch.Tensor

def semantic_scores(class_logits, mask_logits):
    """Native FP32 probability synthesis, excluding the no-object channel."""
    classes = class_logits.float().softmax(-1)[..., :-1]
    masks = mask_logits.float().sigmoid()
    return torch.einsum('bqc,bqhw->bchw', classes, masks)

def normalized_query_embeddings(semantic, output):
    """Apply the native final decoder layer normalization exactly once."""
    hidden = output.transformer_decoder_last_hidden_state
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return semantic.net.model.transformer_module.decoder.layernorm(hidden.float())

def input_valid_mask(height, width, valid_h, valid_w, *, device=None, batch=1):
    if not 0 < valid_h <= height or not 0 < valid_w <= width:
        raise ValueError('Invalid padded-input/content geometry')
    mask = torch.zeros((batch, height, width), dtype=torch.bool, device=device)
    mask[:, :valid_h, :valid_w] = True
    return mask

def resize_adapter_scores(base_scores, delta_scores, padded_hw, valid_hw, target_hw):
    """Restore native FP32 scores and FP64 residuals separately before ranking."""
    def resize(value):
        value = F.interpolate(value, size=padded_hw, mode='bilinear', align_corners=False)
        value = value[:, :, :valid_hw[0], :valid_hw[1]]
        return F.interpolate(value, size=target_hw, mode='bilinear', align_corners=False)
    original = resize(base_scores.float())
    delta = resize(delta_scores.to(torch.float64))
    refined_foreground = original[:, 1:].to(torch.float64) + delta
    return (original, torch.cat((original[:, :1].to(torch.float64), refined_foreground), 1))

def refined_labels(base_scores, refined_scores):
    """Choose material labels while preserving current native foreground support."""
    original = base_scores.argmax(1)
    foreground = refined_scores[:, 1:].to(torch.float64).clamp_min(0.0)
    return torch.where(original > 0, foreground.argmax(1) + 1, 0)

def material_nll(adapted, targets):
    """FP64 conditional material NLL; negative targets are excluded."""
    target = targets.reshape(-1)
    foreground = adapted.refined_scores[:, 1:].permute(0, 2, 3, 1).reshape(-1, 3)
    keep = target >= 0
    if not keep.any():
        return (foreground.sum() * 0.0, 0)
    values = foreground[keep].to(torch.float64).clamp_min(torch.finfo(torch.float64).tiny)
    log_probability = values.log() - values.sum(-1, keepdim=True).log()
    return (F.nll_loss(log_probability, target[keep]), int(keep.sum()))
