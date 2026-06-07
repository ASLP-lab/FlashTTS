"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from random import random, randint 
# import random

from typing import Callable
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
import numpy as np
from scipy.io.wavfile import write
from .ecapa_tdnn import ECAPA_TDNN
from .modules import MelSpec
from .utils import (
    default,
    exists,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)

from functools import partial
from einops import rearrange

# ============================================================================ #
# MeanFlow helpers
# ============================================================================ #

def stopgrad(x):
    """Stop gradient flow."""
    return x.detach()


def adaptive_l2_loss(error, mask, gamma=0., c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||delta||^2. Weights by inverse error so easier
    samples get higher weight. w = 1 / (||delta||^2 + c)^p, p = 1 - gamma.
    """
    delta_sq = torch.mean(error ** 2, dim=-1)  # (B, seq_len)
    delta_sq = delta_sq[mask]  # valid positions only
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||²
    return (stopgrad(w) * loss).mean()



def final_convert_wav(wav_data):
    audio = wav_data.data.cpu().float().numpy()

    audio = audio * 32768.0
    audio = audio.astype(np.int16)
    return audio

def hifigan_convert_wav(wav_data,volume_scale=0.5):
    wav_data = wav_data * volume_scale
    audio = wav_data * 32768.0
    audio = audio.cpu().numpy().astype('int16')
    return audio


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask
        # Contrastive weight (lambda in paper); tune as hyperparameter
        self.contrastive_weight = 0.025 
        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        self.T_hat = torch.load("./osum_dit/T_hat_5k.pt")
        self.T_hat = self.T_hat.unsqueeze(0) 
        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device
    
    def _sample_t_r_pair(
        self, 
        batch_size: int, 
        device: torch.device,
        flow_ratio: float = 0.5,
        time_dist: str = 'lognorm',
        mu: float = -0.4,
        sigma: float = 1.0,
    ):
        """Sample (t, r) for MeanFlow: flow_ratio one-step (r=t), rest multi-step (r<t)."""
        if time_dist == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif time_dist == 'lognorm':
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))
        else:
            raise ValueError(f"Unknown time_dist: {time_dist}")
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])
        num_selected = int(flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]
        t = torch.tensor(t_np, device=device, dtype=torch.float32)
        r = torch.tensor(r_np, device=device, dtype=torch.float32)
        
        return t, r
    def forward_meanflow(
        self,
        inp: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        ref_embed,
        spk_emb,
        flow_ratio: float = 0.5,
        cfg_scale: float = 2.0,
        adaptive_loss_p: float = 0.5,
        use_cfg: bool = True,
        use_jvp: bool = True,
        jvp_api: str = 'autograd',
    ):
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)  # (B, n_mels, T) -> (B, T, n_mels)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device

        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)
        t, r = self._sample_t_r_pair(
            batch_size=batch,
            device=device,
            flow_ratio=flow_ratio,
            time_dist='lognorm',
            mu=-0.4,
            sigma=1.0,
        )
        
        # Broadcast (B,) -> (B, 1, 1)
        t_ = rearrange(t, "b -> b 1 1")
        r_ = rearrange(r, "b -> b 1 1")
        x1 = inp
        x0 = torch.randn_like(x1)
        z = (1 - t_) * x1 + t_ * x0  # noise interpolation
        v = x0 - x1  # target flow
        cond = ref_embed
        # CFG: v_hat = w*v + (1-w)*u_uncond
        
        if use_cfg and cfg_scale != 1.0:
            with torch.no_grad():
                # Unconditional prediction
                u_uncond = self.transformer(
                    x=z,
                    cond=cond,
                    text=text,
                    spk_emb=spk_emb,
                    time=t,
                    t_r=r,
                    drop_audio_cond=True,
                    drop_text=True,
                )
            # CFG: v_hat = w*v + (1-w)*u_uncond
            v_hat = cfg_scale * v + (1.0 - cfg_scale) * u_uncond
        else:
            # No CFG: use target flow directly
            v_hat = v
        # JVP setup
        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            jvp_fn = torch.func.jvp
            create_graph = False
        elif jvp_api == 'autograd':
            jvp_fn = torch.autograd.functional.jvp
            create_graph = True
        # Forward + JVP for u and du/dt
        
        def model_fn(z_in, t_in, r_in):
            return self.transformer(
                x=z_in,
                cond=cond,
                text=text,
                spk_emb=spk_emb,
                time=t_in,
                t_r=r_in,
                drop_audio_cond=False,
                drop_text=False,
            )
        if jvp_api == 'funtorch':
            u, dudt = jvp_fn(
                lambda z, t: model_fn(z, t, r),
                (z, t, r),
                (v_hat, torch.ones_like(t), torch.zeros_like(r)),
            )
        elif jvp_api == 'autograd':
            u, dudt = jvp_fn(
                func=model_fn,
                inputs=(z, t, r),
                v=(v_hat, torch.ones_like(t), torch.zeros_like(r)),
                create_graph=create_graph,
            )

        u_tgt = v_hat - (t_ - r_) * dudt
        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error, mask, gamma=1.0 - adaptive_loss_p)
        with torch.no_grad():
            mse_val = (error ** 2).mean(dim=-1)
            mse_val = mse_val[mask].mean()
        return loss, mse_val, cond, u

    def forward_meanflow_pimproved(
        self,
        inp: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        ref_embed,
        spk_emb,
        flow_ratio: float = 0.5,
        cfg_scale: float = 2.0,
        adaptive_loss_p: float = 0.5,
        use_cfg: bool = True,
        use_jvp: bool = True,
        jvp_api: str = 'autograd',
    ):
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)  # (B, n_mels, T) -> (B, T, n_mels)
            assert inp.shape[-1] == self.num_channels
        
        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device

        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)
        t, r = self._sample_t_r_pair(
            batch_size=batch,
            device=device,
            flow_ratio=flow_ratio,
            time_dist='lognorm',
            mu=-0.4,
            sigma=1.0,
        )
        
        # Broadcast (B,) -> (B, 1, 1)
        t_ = rearrange(t, "b -> b 1 1")
        r_ = rearrange(r, "b -> b 1 1")
        x1 = inp
        x0 = torch.randn_like(x1)
        z = (1 - t_) * x1 + t_ * x0  # noise interpolation
        v = x0 - x1  # target flow
        cond = ref_embed
        # CFG: v_hat = w*v + (1-w)*u_uncond
        
        ## loss for xpred jvp and xpred_mse
        xpred_loss = 0.0
        loss = xpred_loss

        return loss, mse_val, cond, u

    