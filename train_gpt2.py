from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer('NANO_GPT_SCALE_INIT', torch.tensor(1.0))
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head,C//self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head,C//self.n_head).transpose(1,2)
        # att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
    
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.register_buffer('NANO_GPT_SCALE_INIT', torch.tensor(1.0))

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.atten = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.atten(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 
    

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module,'NANO_GPT_SCALE_INIT'):
                std *= (2* self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn : p for pn,p in self.named_parameters()}
        param_dict = {pn : p for pn,p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim()>=2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim()<2]
        optim_group= [
            {'params':decay_params, 'weight_decay':weight_decay},
            {'params':nodecay_params, 'weight_decay':0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_group, lr=learning_rate, betas=(0.9,0.95),eps=1e-8,fused=use_fused)
        return optimizer

    def forward(self, idx, target=None):
        B, T = idx.size()
        assert T<= self.config.block_size, f"Cannot forward sequence of length {T}, becau"
        pos = torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),target.view(-1))
        return logits, loss


from torch.distributed import init_process_group, destroy_process_group
ddp = int(os.environ.get('RANK',-1)) != -1
if ddp:
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = 'mps'
    print(f"using device: {device}")



import tiktoken
import numpy as np
def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataloaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train','val'}
        data_root = ""
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split: {split}"
        if master_process:
            print(f"loaded {len(self.shards)} shards for split: {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B,T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position+=B *T * self.num_processes
        if self.current_position+ (B*T*self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

import time
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

if master_process:
    out_dir = 'out'
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(out_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'train.log')

total_batch_size = 524288           #2**19
B = 16
T = 1024
assert total_batch_size % (B*T*ddp_world_size) == 0, "Make sure total_batch_size is divisible by B*T*ddp_world_size"
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
if master_process:
    print(f"total batch size: {total_batch_size}")
    print(f"gradient accumulation steps: {grad_accum_steps}")
train_loader = DataloaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size, split='train')
val_loader = DataloaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size, split='val')
torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > warmup_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0<= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-7, device= device)

for step in range(max_steps):
    t0 = time.time()
    if step%100 ==0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = torch.zeros(1, device=device)
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x , y = val_loader.next_batch()
                x , y = x.to(device), y.to(device)
                with torch.autocast(device, dtype=torch.bfloat16):
                    logits, loss = model(x,y)
                loss = loss / grad_accum_steps
                val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"Val loss: {val_loss_accum.item():.4f}")
                with open(log_file, 'a') as f:
                    f.write(f"Val loss: {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 5000 ==0 or last_step):
                    checkpoint_path = os.path.join(log_dir, f"checkpoint_{step}.pt")
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    torch.save(checkpoint, checkpoint_path)
                    last_step = step

    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x ,y = x.to(device), y.to(device)
        with torch.autocast(device, dtype=torch.bfloat16):
            logits, loss = model(x,y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / dt
    if master_process:
        print(f"Step {step:4d} | loss: {loss.item()} | norm: {norm} | dt: {dt*1000:.2f}ms, tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()