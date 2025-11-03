from __future__ import annotations

from typing import Iterable, Protocol

import torch
from torch.utils.data import DataLoader


class LanguageModel(Protocol):
    def __call__(self, **kwargs) -> torch.Tensor:  # pragma: no cover - protocol
        ...


def compute_perplexity(model: LanguageModel, dataloader: DataLoader) -> float:
    total_loss = 0.0
    total_tokens = 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    
    # Try to get device from model parameters
    device = torch.device('cpu')
    if hasattr(model, 'parameters'):
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            pass

    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device).reshape(-1)

        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        logits = logits.view(-1, logits.size(-1))

        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        total_tokens += (labels != -100).sum().item()

    if total_tokens == 0:
        return float("inf")

    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def accuracy(predictions: Iterable[int], targets: Iterable[int]) -> float:
    preds = list(predictions)
    targs = list(targets)
    if not preds or len(preds) != len(targs):
        return 0.0

    correct = sum(int(p == t) for p, t in zip(preds, targs))
    return correct / len(preds)


__all__ = ["compute_perplexity", "accuracy"]


