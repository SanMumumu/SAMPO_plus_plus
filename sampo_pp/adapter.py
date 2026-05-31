"""Continuous latent adapter and latent-pyramid construction.

The adapter wraps an external continuous VAE and exposes a continuous latent
pyramid to the rest of the world model. Two backends are supported:

  * ``debug``               - a deterministic, frozen, dependency-free codec used
                              for integration and unit tests. It is *not* meant
                              to reconstruct the input; it only has to satisfy
                              the shape / range / pyramid contracts so the rest
                              of the pipeline can be exercised end to end.
  * ``lightningdit_vavae``  - loads VA-VAE / LightningDiT (see ``_load_vavae``).

The pyramid is always built by recursive average pooling of the finest latent,
matching the paper's Laplacian-style operator ``P_down``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AdapterConfig


class ContinuousLatentAdapter(nn.Module):
    def __init__(self, config: AdapterConfig):
        super().__init__()
        if not isinstance(config, AdapterConfig):
            config = AdapterConfig(**config)
        self.config = config
        self.backend = config.backend
        self.latent_channels = config.latent_channels
        self.latent_scales = tuple(config.latent_scales)
        self.latent_resolution = config.latent_resolution
        self.input_resolution = config.input_resolution

        if self.backend == 'debug':
            self._build_debug_codec()
        elif self.backend == 'lightningdit_vavae':
            self._load_vavae(config)
        else:
            raise ValueError(f'Unknown adapter backend: {self.backend}')

    # ------------------------------------------------------------------ #
    # Backends
    # ------------------------------------------------------------------ #
    def _build_debug_codec(self):
        """Fixed, frozen linear codec (deterministic across runs)."""
        gen = torch.Generator().manual_seed(0)
        enc = torch.randn(self.latent_channels, 3, generator=gen) * 0.5
        dec = torch.randn(3, self.latent_channels, generator=gen) * 0.5
        self.register_buffer('enc_w', enc)
        self.register_buffer('dec_w', dec)

    def _load_vavae(self, config: AdapterConfig):  # pragma: no cover - needs external repo
        import sys

        if config.repo_path:
            sys.path.append(config.repo_path)
        from tokenizer.autoencoder import AutoencoderKL  # type: ignore

        self.vae = AutoencoderKL.from_pretrained(config.ckpt_path)
        self.vae.eval()

    # ------------------------------------------------------------------ #
    # Encode / decode
    # ------------------------------------------------------------------ #
    def encode_frames(self, video: torch.Tensor) -> torch.Tensor:
        """[B, T, 3, H, W] (in [0, 1]) -> [B, T, C, Hl, Wl] continuous latents."""
        B, T, _, H, W = video.shape
        x = video.reshape(B * T, 3, H, W)
        if self.backend == 'debug':
            x = F.adaptive_avg_pool2d(x, self.latent_resolution)
            z = torch.einsum('oc,nchw->nohw', self.enc_w.to(x.dtype), x)
        else:  # pragma: no cover
            x = F.interpolate(x, size=self.config.model_resolution, mode='bilinear', align_corners=False)
            z = self.vae.encode(x * 2 - 1).sample()
            z = F.adaptive_avg_pool2d(z, self.latent_resolution)
        return z.reshape(B, T, self.latent_channels, self.latent_resolution, self.latent_resolution)

    def decode_frames(self, latents: torch.Tensor) -> torch.Tensor:
        """[B, T, C, Hl, Wl] -> [B, T, 3, H, W] reconstructed frames in [0, 1]."""
        B, T, C, h, w = latents.shape
        z = latents.reshape(B * T, C, h, w)
        if self.backend == 'debug':
            x = torch.einsum('oc,nchw->nohw', self.dec_w.to(z.dtype), z)
            x = F.interpolate(x, size=self.input_resolution, mode='bilinear', align_corners=False)
            x = torch.sigmoid(x)
        else:  # pragma: no cover
            z = F.interpolate(z, size=self.config.model_resolution // 8, mode='bilinear', align_corners=False)
            x = self.vae.decode(z).sample
            x = F.interpolate(x, size=self.input_resolution, mode='bilinear', align_corners=False)
            x = (x.clamp(-1, 1) + 1) / 2
        return x.reshape(B, T, 3, self.input_resolution, self.input_resolution)

    # ------------------------------------------------------------------ #
    # Pyramid
    # ------------------------------------------------------------------ #
    def build_pyramid(self, fine_latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Recursive average-pool the finest latent into the full scale pyramid."""
        B, T, C, H, W = fine_latents.shape
        flat = fine_latents.reshape(B * T, C, H, W)
        pyramid = {}
        for s in self.latent_scales:
            pooled = F.adaptive_avg_pool2d(flat, s) if s != H else flat
            pyramid[f'pyramid_{s}'] = pooled.reshape(B, T, C, s, s)
        return pyramid

    def prepare_batch(self, video: torch.Tensor, context_length: int) -> Dict:
        fine = self.encode_frames(video)
        prepared = {'fine_latents': fine, 'context_length': int(context_length)}
        prepared.update(self.build_pyramid(fine))
        return prepared

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def freeze(self) -> 'ContinuousLatentAdapter':
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    def save_pretrained(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'adapter_config.json').write_text(json.dumps(self.config.to_dict()), encoding='utf-8')
        torch.save(self.state_dict(), save_dir / 'adapter_state.pt')

    @classmethod
    def from_pretrained(cls, load_dir) -> 'ContinuousLatentAdapter':
        load_dir = Path(load_dir)
        config = AdapterConfig(**json.loads((load_dir / 'adapter_config.json').read_text(encoding='utf-8')))
        adapter = cls(config)
        state_path = load_dir / 'adapter_state.pt'
        if state_path.exists():
            adapter.load_state_dict(torch.load(state_path, map_location='cpu'), strict=False)
        return adapter
