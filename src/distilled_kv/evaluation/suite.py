from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ..config import EvaluationConfig
from ..logging import get_logger
from ..types import EvalResult
from ..utils.metrics import compute_perplexity


@dataclass
class EvaluationInputs:
    ppl: Optional[torch.utils.data.DataLoader] = None
    mmlu: Optional[torch.utils.data.DataLoader] = None
    gsm8k: Optional[torch.utils.data.DataLoader] = None
    glue: Optional[torch.utils.data.DataLoader] = None


class EvaluationSuite:
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)

    def run(self, model: torch.nn.Module, inputs: EvaluationInputs) -> EvalResult:
        metrics: Dict[str, float] = {}
        details: Dict[str, float] = {}

        model.eval()

        if self.config.run_ppl and inputs.ppl:
            metrics["ppl"] = compute_perplexity(model, inputs.ppl)

        if self.config.run_mmlu and inputs.mmlu:
            metrics["mmlu_acc"] = self._simple_accuracy(model, inputs.mmlu)

        if self.config.run_gsm8k and inputs.gsm8k:
            metrics["gsm8k_acc"] = self._simple_accuracy(model, inputs.gsm8k)

        if self.config.run_glue and inputs.glue:
            metrics["glue_acc"] = self._simple_accuracy(model, inputs.glue)

        self.logger.info("Evaluation metrics: %s", metrics)
        return EvalResult(metrics=metrics, details=details)

    def _simple_accuracy(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> float:
        device = next(model.parameters()).device
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        return correct / max(total, 1)


__all__ = ["EvaluationSuite", "EvaluationInputs"]


