"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, ModuleDict
from einops import rearrange
from ..ecapa_tdnn import ECAPA_TDNN
from x_transformers.x_transformers import RotaryEmbedding
from torch.amp import autocast
from ..modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNorm_Final,
    # AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)

def exists(val):
    return val is not None

# ---------------------------
# Text embedding module
# ---------------------------
class TextEmbedding(nn.Module):
    """Map text tokens to embeddings (vocab size + 1 for filler)."""
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, align_mode: str = "repeat"):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        self.align_mode = align_mode

    def forward(self, text: torch.Tensor, seq_len: int, drop_text: bool = False) -> torch.Tensor:
        if drop_text:
            text = torch.zeros_like(text)
        x = self.text_embed(text)
        x = torch.repeat_interleave(x, repeats=4, dim=1)
        current_len = x.shape[1]
        if current_len < seq_len:
            pad_len = seq_len - current_len
            padding = torch.zeros(x.size(0), pad_len, x.size(2), 
                                dtype=x.dtype, device=x.device)
            x = torch.cat([x, padding], dim=1)  # [batch, seq_len, embed_dim]
        elif current_len > seq_len:
            x = x[:, :seq_len, :]
        return x

# ---------------------------
# Input embedding: fuse audio and text conditions
# ---------------------------
class InputEmbedding(nn.Module):
    """Fuse noised audio, cond audio, and text embedding for the transformer."""
    def __init__(self, mel_dim: int, text_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(mel_dim + text_dim + 128 + 192, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)
        self.spk_encoder = ECAPA_TDNN(
                                input_size=80,
                                lin_neurons=128,
                                activation=torch.nn.ReLU,
                                channels=[512, 512, 512, 512, 1536],
                                kernel_sizes=[5, 3, 3, 3, 1],
                                dilations=[1, 2, 3, 4, 1],
                                attention_channels=128,
                                res2net_scale=8,
                                se_channels=128,
                                global_context=True,
                            )

    def forward(
        self, 
        x: torch.Tensor,
        cond: torch.Tensor,
        spk_emb: torch.Tensor,
        text_embed: torch.Tensor,
        drop_audio_cond: bool = False,
    ) -> torch.Tensor:
        """Fuse x, cond, text_embed; zero cond/spk_emb when drop_audio_cond (CFG)."""
        try:
            if drop_audio_cond:
                cond = torch.zeros((x.shape[0], 128), dtype=x.dtype, device=x.device)
                spk_emb = torch.zeros_like(spk_emb)
            else:
                cond = self.spk_encoder(cond)
            
            #[bs,192]
            cond = cond.unsqueeze(1).repeat(1, x.size(1), 1)
            spk_emb = spk_emb.repeat(1, x.size(1), 1)
            # cond = cond[:, None, :].expand(-1, x.shape[1], -1)
            # Concat x, cond, text_embed on last dim
            merged = torch.cat((x, cond, text_embed,spk_emb), dim=-1)
            # Linear to out_dim
            x = self.proj(merged)
            
            # Conv position embedding
            # x = self.conv_pos_embed(x) + x
        except Exception as e:
            print(e)

        return x

    def fast_forward(self, x, cond, spk_emb, text_embed, text_embed_uncond):
        x = torch.cat([x,x],dim=0)
        spk = spk_emb.repeat(1, x.size(1), 1)
        spk = torch.cat([spk,torch.zeros_like(spk)],dim=0)
        cond = self.spk_encoder(cond)
        # cond = cond.unsqueeze(1).repeat(1, x.size(1), 1)
        cond = torch.cat([cond,torch.rand_like(cond)],dim=0)
        cond = cond.unsqueeze(1).repeat(1, x.size(1), 1)
        # cond = self.spk_encoder(cond).unsqueeze(1).repeat(1, x.size(1), 1)
        text_emb = torch.cat([text_embed,text_embed_uncond],dim=0)
        # print(x.shape,cond.shape,text_embed.shape)
        x = self.proj(torch.cat((x, cond, text_emb, spk), dim=-1))

        return x


# ---------------------------
# DiT: Transformer backbone (DiTBlock-based)
# ---------------------------
class DiT(nn.Module):
    """DiT: time + text + audio conditions, block/local attention."""
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=80,
        text_num_embeds=6562,
        text_dim=None,
        text_mask_padding=True,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        long_skip_connection=False,
        checkpoint_activations=False,
        forward_layers=[0],
        backward_layers=[5, 10, 15],
    ):
        super().__init__()
        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers, align_mode="interpolate")
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        # Forward/backward layer indices
        forward_layers = set(forward_layers) if forward_layers else set()
        backward_layers = set(backward_layers) if backward_layers else set()

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(
                dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout,
                block_size=24,
                t_p=1 if i in backward_layers else 0,  # backward
                t_f=1 if i in forward_layers else 0    # forward
            ) for i in range(depth)
        ])
        
        # Long skip: concat input and output then project
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        # Final norm (AdaLayerNormZero_Final)
        self.norm_out = AdaLayerNorm_Final(dim)
        # Output projection to mel
        self.proj_out = nn.Linear(dim, mel_dim)

        # Optional activation checkpointing
        self.checkpoint_activations = checkpoint_activations #False

    def ckpt_wrapper(self, module):
        """
        Wrap for activation checkpointing (see fast-DiT).
        """
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    
    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None
        
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        spk_emb: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        drop_audio_cond,
        drop_text,
        mask: torch.BoolTensor | None = None,
        chunk_offset: int = 0,
    ) -> torch.Tensor:
        """Forward: time -> text -> input embed -> rope -> DiT blocks -> norm -> mel."""
        batch, seq_len = x.shape[0], x.shape[1]
        
        # Expand time to batch
        if time.ndim == 0:
            time = time.repeat(batch)
        
        # Time embedding
        t = self.time_embed(time)
        
        
        # Text embedding
        text_embed = self.text_embed(text, seq_len, drop_text=drop_text)

        x = self.input_embed(x, cond, spk_emb, text_embed ,drop_audio_cond=drop_audio_cond)

        # Rotary embedding
        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        # Residual for long skip
        if self.long_skip_connection is not None: #none
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # checkpoint
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope)
            else:
                x = block(x, t, mask=mask, rope=rope)
            # x = block(x, t, mask=mask, rope=rope)

        # Long skip fusion
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        # Norm
        x = self.norm_out(x, t)
        # Output to mel
        output = self.proj_out(x) #Linear(in_features=1024, out_features=80, bias=True)

        return output

    def fast_forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        spk_emb: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        t_r: torch.Tensor,
        drop_audio_cond,
        drop_text,
        mask: torch.BoolTensor | None = None,
        chunk_offset: int = 0,
    ) -> torch.Tensor:
        """Fast forward (cond + uncond stacked)."""
        batch, seq_len = x.shape[0] * 2, x.shape[1]
        
        if time.ndim == 0:
            time = time.repeat(batch)
        
        # Time embedding
        t = self.time_embed(time) #[bs,time]
        t_r = self.time_embed(t_r) #[bs,time]
        
        t = t + t_r
        text_embed = self.text_embed(text, seq_len, drop_text=False)
        text_embed_uncond = self.text_embed(text, seq_len, drop_text=True)
        
        # Input embed (cond + uncond)
        
        # x = self.input_embed(x, cond, spk_emb, text_embed ,drop_audio_cond=drop_audio_cond)
        # pos  = torch.arange(start=chunk_offset,end=chunk_offset+seq_len, device = x.device)
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        x = self.input_embed.fast_forward(x, cond, spk_emb, text_embed, text_embed_uncond)
        # Rotary embedding
        # rope from chunk_offset for streaming
        
        # Residual for long skip
        if self.long_skip_connection is not None: #none
            residual = x

        for block in self.transformer_blocks:
            # if self.checkpoint_activations:
            #     x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope)
            # else:
            #     x = block(x, t, mask=mask, rope=rope)
            x = block(x, t, mask=mask, rope=rope)

        # Long skip fusion
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        # Norm
        x = self.norm_out(x, t)
        # Output to mel
        output = self.proj_out(x) #Linear(in_features=1024, out_features=80, bias=True)

        return output