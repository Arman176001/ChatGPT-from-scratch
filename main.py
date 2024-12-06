with open('input.txt','r',encoding='utf-8') as f:
  text = f.read()

import torch
import torch.nn as nn
from torch.nn import functional as F

chars = sorted(list(set(text)))
vocab_size = len(chars)
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
batch_size = 64 
block_size = 256
max_iters = 5000
eval_interval = 500
eval_iters = 200
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)
#simple tokenizer    famous tokenizer-> sentencePiece(sub word)(google)  tiktoken(bi pair encoding)(openAI)
#making encoder and decoder
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # takes a string and outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # takes a list of numbers and outputs characters

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data)-block_size,(batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x = x.to(device)
  y = y.to(device)
  return x, y

class Head(nn.Module):
  """one head of self attention"""
  def __init__(self,head_size):
    super().__init__()
    self.key = nn.Linear(n_embd,head_size,bias=False)
    self.query = nn.Linear(n_embd,head_size,bias=False)
    self.value = nn.Linear(n_embd,head_size,bias=False)
    self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)

    wei = q @ k.transpose(-2,-1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
    wei = F.softmax(wei,dim=-1)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v
    return out
  
#MultiheadAttention

class MultiheadAttention(nn.Module):
  def __init__(self,num_heads,head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd,n_embd)
    self.dropout = nn.Dropout(dropout)
  def forward(self,x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    return out

class FeedForward(nn.Module):
  def __init__(self,n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout),
    )

  def forward(self,x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self,n_embd,n_head):
    super().__init__()
    head_size = n_embd//n_head
    self.sa = MultiheadAttention(n_head,head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self,x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train' , 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      X, Y = X.to(device), Y.to(device)
      logits, loss = model(X,Y)
      losses[k] = loss.item()
    out[split] = loss.mean()
  model.train()
  return out

#Bigram Language Model with multihead attention
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) 
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self,idx,targets=None):
    B,T = idx.shape
    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T,device=device))
    x = tok_emb + pos_emb
    x = self.blocks(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits,targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:,-block_size:]
      logits, loss = self(idx_cond)
      logits = logits[:,-1,:]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next),dim=1)
    return idx
  
model= BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
for steps in range(max_iters):
  if steps % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
  xb, yb = get_batch('train')
  #evaluate loss
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

print(decode(model.generate(idx = torch.zeros((1,1),dtype=torch.long), max_new_tokens=1000)[0].tolist()))