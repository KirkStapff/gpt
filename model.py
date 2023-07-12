import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx)  # (B,T,C)
        B, T, C = logits.shape

        if targets is None:
            return logits, None

        targets = targets.view(B*T)
        loss = F.cross_entropy(logits.view(B*T, C), targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # (B, C) last T
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
