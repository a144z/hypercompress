from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn


@dataclass
class ModelBundle:
    """Container holding teacher and student models plus optional tokenizer."""

    teacher: nn.Module
    student: nn.Module
    tokenizer: Optional[Any] = None
    label: str = ""


@dataclass
class BaselineMetrics:
    """Reference metrics computed prior to compression."""

    perplexity: Optional[float] = None
    mmlu: Optional[float] = None
    additional: Dict[str, float] = field(default_factory=dict)


@dataclass
class BranchArtifact:
    """Result emitted by an individual compression branch."""

    name: str
    state_dict: Dict[str, torch.Tensor]
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class BranchPlan:
    """Descriptor for how a branch should run against the current models."""

    name: str
    parameters: Dict[str, Any]
    enabled: bool = True


@dataclass
class PipelineState:
    """Mutable state passed through the pipeline."""

    bundle: ModelBundle
    work_dir: Path
    device: torch.device
    baselines: BaselineMetrics = field(default_factory=BaselineMetrics)
    branch_results: Dict[str, BranchArtifact] = field(default_factory=dict)
    merged_state: Optional[Dict[str, torch.Tensor]] = None
    distillation_history: Dict[str, Any] = field(default_factory=dict)
    finetune_history: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Structured evaluation output."""

    metrics: Dict[str, float]
    details: Dict[str, Any] = field(default_factory=dict)


def iter_named_parameters(model: nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    """Return an iterable over parameters ensuring tensors are detached copies."""

    for name, param in model.named_parameters():
        if not isinstance(param, torch.Tensor):
            continue
        yield name, param.detach().clone()


__all__ = [
    "ModelBundle",
    "BaselineMetrics",
    "BranchArtifact",
    "BranchPlan",
    "PipelineState",
    "EvalResult",
    "iter_named_parameters",
]


