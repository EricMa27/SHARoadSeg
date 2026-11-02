"""Shared shoulder-aware attention at stride-8 decoder layers 3, 6, and 9."""
from __future__ import annotations
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F
INSERTION_LAYERS = (2, 5, 8)
SHA_SEED = 4203
HALO_MASS_EPS = 0.001

def soft_collar_prior(mask_logits, valid_mask, feature_size):
    """Detached normalized max-pool collar; padding and negligible mass excluded."""
    if mask_logits.ndim != 4 or valid_mask.ndim != 3:
        raise ValueError('Expected mask_logits[B,Q,h,w] and valid_mask[B,H,W]')
    if mask_logits.shape[0] != valid_mask.shape[0]:
        raise ValueError('Mask and validity batch sizes differ')
    with torch.autocast(device_type=mask_logits.device.type, enabled=False):
        logits = F.interpolate(mask_logits.detach().float(), size=feature_size, mode='bilinear', align_corners=False)
        probability = logits.sigmoid()
        coverage = F.interpolate(valid_mask[:, None].detach().float(), size=feature_size, mode='area')
        valid = coverage >= 1.0 - 1e-06
        pooled = F.max_pool2d(probability.masked_fill(~valid, -torch.inf), kernel_size=5, stride=1, padding=2)
        halo = torch.where(valid & torch.isfinite(pooled), (pooled - probability).clamp_min(0.0), torch.zeros_like(probability)).flatten(2)
        mass = halo.sum(-1)
        active = mass > HALO_MASS_EPS
        prior = torch.where(active[..., None], halo / mass[..., None].clamp_min(1e-12), torch.zeros_like(halo))
    return (prior, mass)

class SharedLocalCollarAttention(nn.Module):
    """Four-head attention with log spatial prior and a learned zero-valued null."""

    def __init__(self, feature_dim=256, attention_dim=128, num_heads=4):
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError('attention_dim must be divisible by num_heads')
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.query = nn.Linear(feature_dim, attention_dim, bias=False)
        self.key = nn.Linear(feature_dim, attention_dim, bias=False)
        self.value = nn.Linear(feature_dim, attention_dim, bias=False)
        self.output = nn.Linear(attention_dim, feature_dim, bias=False)
        self.null_key = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.null_bias = nn.Parameter(torch.zeros(num_heads))
        nn.init.zeros_(self.output.weight)

    def _heads(self, sequence):
        length, batch, _ = sequence.shape
        return sequence.reshape(length, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

    def project_memory(self, key_with_position, value):
        if key_with_position.shape != value.shape:
            raise ValueError('Key and value memory shapes differ')
        return (self._heads(self.key(key_with_position)), self._heads(self.value(value)))

    def forward(self, query_with_position, projected_memory, prior, collect_diagnostics=False):
        key, value = projected_memory
        query = self._heads(self.query(query_with_position))
        if prior.shape != (query.shape[0], query.shape[2], key.shape[2]):
            raise ValueError('Soft-collar prior does not match query/memory')
        with torch.autocast(device_type=query.device.type, enabled=False):
            query_f, key_f, value_f = (query.float(), key.float(), value.float())
            scale = 1.0 / math.sqrt(self.head_dim)
            spatial = torch.matmul(query_f, key_f.transpose(-1, -2)) * scale
            log_prior = prior.float().clamp_min(1e-30).log()
            log_prior = log_prior.masked_fill(prior <= 0, -torch.inf)
            spatial = spatial + log_prior[:, None]
            null = (query_f * self.null_key.float()[None, :, None]).sum(-1)
            null = null * scale + self.null_bias.float()[None, :, None]
            attention = torch.cat((spatial, null[..., None]), -1).softmax(-1)
            context = torch.matmul(attention[..., :-1], value_f)
            context = context.permute(2, 0, 1, 3).reshape(query_with_position.shape[0], query_with_position.shape[1], self.attention_dim)
        update = self.output(context.to(query_with_position.dtype))
        diagnostics = attention[..., -1].detach() if collect_diagnostics else None
        return (update, diagnostics)

class ShoulderAwareDecoder(nn.Module):
    """Native decoder integration used by the complete SHARoadSeg model."""

    def __init__(self, semantic):
        super().__init__()
        self.semantic = semantic
        self.sha_enabled = True
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(SHA_SEED)
            self.sha = SharedLocalCollarAttention()
        self.sha.to(next(semantic.parameters()).device)
        decoder = semantic.net.model.transformer_module.decoder
        if len(decoder.layers) != 9 or decoder.num_feature_levels != 3:
            raise ValueError('SHA requires a nine-layer/three-scale decoder')
        if decoder.layerdrop != 0:
            raise ValueError('SHA requires zero decoder layerdrop')
        if getattr(semantic.net, 'is_gradient_checkpointing', False):
            raise ValueError('SHA does not support gradient checkpoint replay')
        if int(semantic.net.config.hidden_dim) != 256:
            raise ValueError('SHA requires native hidden_dim=256')
        self._active_forward = False
        self._collect_diagnostics = False
        self._diagnostics = []
        self.last_insertion_layers = ()
        self.last_mask_prediction_count = 0
        self.last_kv_projection_count = 0
        self._clear_context()
        decoder.register_forward_pre_hook(self._capture_decoder_input, with_kwargs=True)
        decoder.mask_predictor.register_forward_hook(self._capture_mask_prediction)
        for index, hook in zip(INSERTION_LAYERS, (self._inject_layer2, self._inject_layer5, self._inject_layer8)):
            decoder.layers[index].cross_attn.register_forward_hook(hook, with_kwargs=True)

    def _clear_context(self):
        self._valid_mask = None
        self._feature_size = None
        self._memory_value = None
        self._memory_key = None
        self._projected_memory = None
        self._previous_mask_logits = None
        self._mask_prediction_count = 0
        self._kv_projection_count = 0
        self._insertions = []

    def context_is_clear(self):
        return not self._active_forward and self._valid_mask is None and (self._feature_size is None) and (self._memory_value is None) and (self._memory_key is None) and (self._projected_memory is None) and (self._previous_mask_logits is None)

    def _capture_decoder_input(self, decoder, args, kwargs):
        if not self.sha_enabled:
            return
        if not self._active_forward or self._valid_mask is None:
            raise RuntimeError('Call SHA through wrapper.forward, not semantic.net directly')
        if getattr(decoder, 'gradient_checkpointing', False):
            raise RuntimeError('Gradient checkpoint replay is unsupported by SHA')

        def argument(name, index):
            return kwargs[name] if name in kwargs else args[index]
        sizes = argument('feature_size_list', 5)
        memories = argument('encoder_hidden_states', 3)
        positions = argument('multi_stage_positional_embeddings', 1)
        self._feature_size = tuple(sizes[2])
        self._memory_value = memories[2]
        self._memory_key = memories[2] + positions[2]

    def _capture_mask_prediction(self, module, args, output):
        if not self.sha_enabled:
            return
        if not self._active_forward:
            raise RuntimeError('Missing SHA native forward context')
        self._previous_mask_logits = output[0].detach()
        self._mask_prediction_count += 1

    def _inject_layer2(self, module, args, kwargs, result):
        return self._inject(2, module, args, kwargs, result)

    def _inject_layer5(self, module, args, kwargs, result):
        return self._inject(5, module, args, kwargs, result)

    def _inject_layer8(self, module, args, kwargs, result):
        return self._inject(8, module, args, kwargs, result)

    def _inject(self, index, module, args, kwargs, result):
        if not self.sha_enabled:
            return result
        if not self._active_forward or self._previous_mask_logits is None:
            raise RuntimeError('Missing same-forward mask/context for SHA')
        if self._mask_prediction_count != index + 1:
            raise RuntimeError('SHA must read the immediately preceding layer mask')
        query = kwargs.get('query', args[0] if len(args) > 0 else None)
        value = kwargs.get('value', args[2] if len(args) > 2 else None)
        if query is None or value is not self._memory_value:
            raise RuntimeError('SHA insertion is not the shared stride-8 memory')
        if self._projected_memory is None:
            self._projected_memory = self.sha.project_memory(self._memory_key, self._memory_value)
            self._kv_projection_count += 1
        prior, mass = soft_collar_prior(self._previous_mask_logits, self._valid_mask, self._feature_size)
        update, null_probability = self.sha(query, self._projected_memory, prior, collect_diagnostics=self._collect_diagnostics)
        self._insertions.append(index)
        if self._collect_diagnostics:
            self._diagnostics.append(dict(layer=index, null_probability=null_probability, halo_mass=mass.detach()))
        return (result[0] + update.to(result[0].dtype), *result[1:])

    @contextmanager
    def collect_null_diagnostics(self, enabled=True):
        previous = self._collect_diagnostics
        self._collect_diagnostics = bool(enabled)
        try:
            yield self
        finally:
            self._collect_diagnostics = previous

    def get_null_diagnostics(self, cpu=False):
        return tuple(({key: value.cpu() if cpu and torch.is_tensor(value) else value for key, value in row.items()} for row in self._diagnostics))

    def forward(self, images, valid_mask, mask_labels=None, class_labels=None):
        if not self.context_is_clear():
            raise RuntimeError('Nested forward/stale SHA context is unsupported')
        if valid_mask.shape != (images.shape[0], *images.shape[-2:]):
            raise ValueError('valid_mask must match the padded image extent')
        self._active_forward = True
        self._valid_mask = valid_mask if self.sha_enabled else None
        self._diagnostics = []
        try:
            output = self.semantic.net(pixel_values=images, mask_labels=mask_labels, class_labels=class_labels, output_hidden_states=True, return_dict=True)
            if self.sha_enabled and (tuple(self._insertions) != INSERTION_LAYERS or self._mask_prediction_count != 10 or self._kv_projection_count != 1):
                raise RuntimeError('Expected three SHA insertions in one native forward')
        finally:
            self.last_insertion_layers = tuple(self._insertions)
            self.last_mask_prediction_count = self._mask_prediction_count
            self.last_kv_projection_count = self._kv_projection_count
            self._clear_context()
            self._active_forward = False
        return output
