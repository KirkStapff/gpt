import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionHead(nn.Module):

    def __init__(self, vocab_size, head_size, block_size):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.key = nn.Linear(vocab_size, head_size, bias=False)
        self.query = nn.Linear(vocab_size, head_size, bias=False)
        self.value = nn.Linear(vocab_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        probs = wei @ v
        return probs


class MultiHeadAttention(nn.Module):

    def __init__(self, vocab_size, head_size, head_count, block_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(vocab_size=vocab_size, head_size=head_size, block_size=block_size) for _ in range(head_count)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class LanguageModel(nn.Module):

    def __init__(self, vocab_size, embed_size, block_size, batch_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.block_size = block_size
        self.batch_size = batch_size
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)
        self.attention_head = AttentionHead(embed_size, block_size, vocab_size)
        self.final_norm = nn.LayerNorm(embed_size)
        self.model_head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx, targets=None):
        idx = idx[:, -self.block_size:]
        token_embed = self.token_embedding_table(idx)
        position_embed = self.position_embedding_table(torch.arange(self.block_size))

        x = token_embed + position_embed
        x = self.attention_head(x)
        x = self.final_norm(x)
        logits = self.model_head(x)

        if targets is None:
            return logits, None

        targets = targets.view(self.batch_size*self.block_size)
        loss = F.cross_entropy(logits.view(self.batch_size*self.block_size, self.vocab_size), targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx = idx[:, -self.block_size:]
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # (B, C) last T
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
