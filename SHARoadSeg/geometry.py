"""Aspect-preserving input resize and right/bottom padding."""
from __future__ import annotations
import torch
from torch.nn import functional as F
PAD_STRIDE = 32

def letterbox(x: torch.Tensor, long_side: int | None, stride: int=PAD_STRIDE):
    h0, w0 = (int(x.shape[-2]), int(x.shape[-1]))
    if long_side:
        scale = float(long_side) / float(max(h0, w0))
        nh = max(1, int(round(h0 * scale)))
        nw = max(1, int(round(w0 * scale)))
        if (nh, nw) != (h0, w0):
            x = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)
    else:
        nh, nw = (h0, w0)
    ph, pw = (-nh % stride, -nw % stride)
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    return (x, nh, nw)
