import torch.nn as nn
from typing import Any
from mamba_block import MambaBlock
import torch

class MambaBackbone(nn.Module):
    def __init__(self, dim: int, n_layers: int, vocab_size: int, rms_norm: bool = True, norm_eps: float = 1e-5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([MambaBlock(dim, rms_norm=rms_norm, layer_idx=i) for i in range(n_layers)])
        if rms_norm:
            self.norm_f = nn.LayerNorm(dim, eps=norm_eps)

    def forward(self, input_ids: torch.Tensor, inference_params=None) -> Any:
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)
        residual = hidden_states + residual if residual is not None else hidden_states
        hidden_states = self.norm_f(residual)
        return hidden_states