# GPT From Scratch

A minimal, educational implementation of GPT-2-style language models in PyTorch, inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT). This project is designed for learning and experimentation, with clear code and multiple training and inference scripts.

## Features

- **Custom GPT-2 Model**: See [`train_gpt2.py`](train_gpt2.py) and [`train_gpt2Kaggle.py`](train_gpt2Kaggle.py) for a full transformer implementation, including:
  - Multi-head self-attention ([`CasualSelfAttention`](train_gpt2.py))
  - Transformer blocks ([`Block`](train_gpt2.py))
  - Layer normalization, MLP, and tied embeddings
  - Configurable model hyperparameters ([`GPTConfig`](train_gpt2.py))
- **Distributed Training**: Multi-GPU support via PyTorch DDP ([`train_gpt2Kaggle.py`](train_gpt2Kaggle.py))
- **Efficient Data Loading**: Custom dataloader for batching and tokenization ([`DataloaderLite`](train_gpt2.py))
- **Checkpointing**: Save and resume training from checkpoints
- **Text Generation**: Generate text with trained models ([`generate.py`](generate.py))
- **Bigram Baseline**: Simple bigram language model for comparison ([`main.py`](main.py))
- **Tokenizer**: Uses [tiktoken](https://github.com/openai/tiktoken) for fast GPT-2 compatible tokenization

## File Overview

- [`train_gpt2.py`](train_gpt2.py): Main GPT-2 training script (single/multi-GPU)
- [`train_gpt2Kaggle.py`](train_gpt2Kaggle.py): Kaggle/colab-friendly DDP training
- [`generate.py`](generate.py): Script for generating text from a trained checkpoint
- [`main.py`](main.py): Bigram model and playground for basic language modeling
- `t8.shakespeare.txt`: Example training data (Shakespeare)
- [`checkpoint_5000.pt`](out/logs/checkpoint_5000.pt): Example model checkpoint

## Example Usage

**Train a GPT-2 model:**
```sh
train_gpt2.py
```
**Generate text:**
```sh
generate.py
```
**Train on Kaggle/Colab with DDP:**
```sh
train_gpt2Kaggle.py
```
**Credits**

- Core architecture and training loop inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).
- Tokenization via [tiktoken](https://github.com/openai/tiktoken).
