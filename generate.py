from dataclasses import dataclass
import torch
from collections import OrderedDict
from train_gpt2 import GPT, GPTConfig
from torch.nn import functional as F
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

checkpoint = torch.load('out/logs/checkpoint_5000.pt', map_location='cpu')

# Set model config - make sure these match your training setup
@dataclass
class GPTConfig:
    block_size: int = 128      
    vocab_size: int = 50257      
    n_layer: int = 2           
    n_head: int = 2            
    n_embd: int = 128  
model = GPT(GPTConfig(vocab_size=50304))

# Clean state dict keys if needed (remove 'module.' prefix)
model_state_dict = checkpoint['model']
new_state_dict = OrderedDict()
for k, v in model_state_dict.items():
    name = k.replace('module.', '')
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def sample(model, start_tokens, steps, temperature=1.0, top_k=None):
    model.eval()
    start = start_tokens.to(next(model.parameters()).device)
    for _ in range(steps):
        idx_cond = start[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, ix = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        start = torch.cat((start, next_token), dim=1)
    return start

prompt = "Love"
start_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

output_tokens = sample(model, start_tokens, steps=200, temperature=2.0, top_k=40)

output_text = tokenizer.decode(output_tokens[0].tolist())

print("Generated text:\n", output_text)
