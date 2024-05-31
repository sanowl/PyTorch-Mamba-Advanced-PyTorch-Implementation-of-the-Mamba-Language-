# generate.py

import torch

def generate(model, tokenizer, prompt, n_tokens_to_gen, device):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        for _ in range(n_tokens_to_gen):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return output_text
