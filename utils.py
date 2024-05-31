import torch
from typing import Dict

def fetch_weights(model_name: str) -> Dict[str, torch.Tensor]:
    if model_name not in MODELS:
        raise ValueError(f"Requested unknown mamba model: {model_name}")
    # Download weights from Hugging Face
    url = f"https://huggingface.co/state-spaces/mamba-{model_name}/resolve/main/pytorch_model.bin"
    weights = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    return weights