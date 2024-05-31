import argparse
import time
from transformers import AutoTokenizer
from mamba_model import Mamba
from generate import generate

MODELS = {
    "130m": {"dim": 768, "n_layers": 24, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
    "370m": {"dim": 1024, "n_layers": 48, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
    "790m": {"dim": 1536, "n_layers": 48, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
    "1.4b": {"dim": 2048, "n_layers": 48, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
    "2.8b": {"dim": 2560, "n_layers": 64, "vocab_size": 50277, "pad_vocab_size_multiple": 8},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mamba in PyTorch", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prompt", type=str, default="Why is gravity ", help="Prompt for LLM completion")
    parser.add_argument("--size", type=str, default="370m",
                        help=f"Size of model to use [{', '.join([k for k in MODELS.keys()])}]")
    parser.add_argument("--n_tokens", type=int, default=10, help="Number of tokens to generate")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = Mamba.from_pretrained(args.size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    prompt = args.prompt
    num_toks = args.n_tokens
    
    s = time.time()
    output = generate(model, tokenizer, prompt, n_tokens_to_gen=num_toks)
    print(output)
    print('TIME: ', time.time() - s)
    
    TORCHOUTPUT = "Why is gravity \nso important?\nBecause it's the only"
    print('Outputs Match:', output == TORCHOUTPUT)