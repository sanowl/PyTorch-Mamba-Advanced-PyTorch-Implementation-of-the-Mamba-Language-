# mamba_block.py
import torch.nn as nn
from typing import Optional
from mamba_mixer import MambaMixer

class MambaBlock(nn.Module):
    def __init__(self, dim: int, norm_eps: float = 1e-5, rms_norm: bool = True, layer_idx: Optional[int] = None):
        super().__init__()
        self.mixer = MambaMixer(dim, layer_idx=layer_idx)
        if rms_norm:
            self.norm = nn.LayerNorm(dim, eps=norm_eps)
        else:
            raise NotImplementedError

    def forward(self, hidden_states, residual=None, inference_params=None):
        residual = hidden_states + residual if residual is not None else hidden_states
        hidden_states = self.norm(residual)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual
