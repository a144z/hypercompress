from __future__ import annotations

import torch


class TinyByteLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_size: int = 128) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.transform = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(input_ids)
        hidden = self.transform(hidden)
        return self.lm_head(hidden)


__all__ = ["TinyByteLM"]


