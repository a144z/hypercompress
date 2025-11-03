from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import FinetuneConfig
from ..logging import get_logger


@dataclass
class FinetuneReport:
    history: Dict[str, list[float]]
    steps: int


class FineTuner:
    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    def run(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device | None = None,
        progress: Optional[object] = None,
        task: Optional[object] = None,
    ) -> FinetuneReport:
        model.train()
        history = {"loss": [], "ppl": []}

        # Determine device from model if not provided
        if device is None:
            device = next(model.parameters()).device

        best_loss = float("inf")
        patience = self.config.patience
        no_improve = 0

        step_tokens = 0

        for step, batch in enumerate(dataloader, start=1):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)

            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history["loss"].append(loss.item())
            history["ppl"].append(torch.exp(loss.detach()).item())

            if progress and task is not None:
                progress.update(task, advance=1)

            step_tokens += labels.numel()

            if loss.item() + self.config.ppl_tolerance < best_loss:
                best_loss = loss.item()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                self.logger.info("Early stopping triggered after %d steps", step)
                break

            if step_tokens >= self.config.max_tokens:
                self.logger.info("Reached token budget %d", step_tokens)
                break

        return FinetuneReport(history=history, steps=len(history["loss"]))


__all__ = ["FineTuner", "FinetuneReport"]


