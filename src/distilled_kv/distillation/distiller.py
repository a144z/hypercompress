from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import DistillationConfig
from ..logging import get_logger
from ..types import PipelineState


@dataclass
class DistillationReport:
    losses: Dict[str, float]
    steps: int


class KnowledgeDistiller:
    def __init__(self, config: DistillationConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    def distill(
        self,
        state: PipelineState,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        progress: Optional[object] = None,
        task: Optional[object] = None,
    ) -> DistillationReport:
        teacher = state.bundle.teacher
        student = state.bundle.student

        teacher.eval()
        student.train()

        kl_losses: list[float] = []
        activation_losses: list[float] = []

        for step, batch in enumerate(dataloader, start=1):
            inputs = {k: v.to(state.device) for k, v in batch.items() if k != "labels"}
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(state.device)

            with torch.no_grad():
                teacher_outputs = teacher(**inputs)
                teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, "logits") else teacher_outputs

            student_outputs = student(**inputs)
            student_logits = student_outputs.logits if hasattr(student_outputs, "logits") else student_outputs
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)

            kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            activation = F.mse_loss(student_logits, teacher_logits)

            loss = self.config.kl_weight * kl + self.config.activation_weight * activation

            if labels is not None:
                label_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
                loss = loss + label_loss * 0.1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            kl_losses.append(kl.item())
            activation_losses.append(activation.item())

            if progress and task is not None:
                progress.update(task, advance=1)

            if step * student_logits.size(0) >= self.config.token_budget:
                break

        metrics = {
            "kl": float(sum(kl_losses) / max(len(kl_losses), 1)),
            "activation": float(sum(activation_losses) / max(len(activation_losses), 1)),
        }

        self.logger.info("Distillation completed: %s", metrics)
        return DistillationReport(losses=metrics, steps=len(kl_losses))


__all__ = ["KnowledgeDistiller", "DistillationReport"]


