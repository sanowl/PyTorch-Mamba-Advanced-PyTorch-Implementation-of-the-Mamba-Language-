import torch
from typing import Dict

MODELS = {
    "130m": {"dim": 768, "n_layers": 24, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
    "370m": {"dim": 1024, "n_layers": 48, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
    "790m": {"dim": 1536, "n_layers": 48, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
    "1.4b": {"dim": 2048, "n_layers": 48, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
    "2.8b": {"dim": 2560, "n_layers": 64, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
}

def fetch_weights(model_name: str) -> Dict[str, torch.Tensor]:
    if model_name not in MODELS:
        raise ValueError(f"Requested unknown mamba model: {model_name}")
    # Download weights from Hugging Face
    url = f"https://huggingface.co/state-spaces/mamba-{model_name}/resolve/main/pytorch_model.bin"
    weights = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    return weights