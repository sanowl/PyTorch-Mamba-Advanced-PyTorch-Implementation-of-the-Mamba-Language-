import torch.nn as nn
from mamba_backbone import MambaBackbone
from utils import fetch_weights

MODELS = {
    "130m": {"dim": 768, "n_layers": 24, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
    "370m": {"dim": 1024, "n_layers": 48, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
    "790m": {"dim": 1536, "n_layers": 48, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
    "1.4b": {"dim": 2048, "n_layers": 48, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
    "2.8b": {"dim": 2560, "n_layers": 64, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
}

class Mamba(nn.Module):
    def __init__(self, dim: int, n_layers: int, vocab_size: int, pad_vocab_size_multiple: int = 1):
        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MambaBackbone(dim, n_layers, vocab_size)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, inference_params=None, num_last_tokens=0):
        hidden_states = self.backbone(input_ids, inference_params=inference_params)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        return self.lm_head(hidden_states)

    @staticmethod
    def from_pretrained(model_name: str):
        weights = fetch_weights(model_name)
        model = Mamba(**MODELS[model_name])
        model.load_state_dict(weights)
        return model