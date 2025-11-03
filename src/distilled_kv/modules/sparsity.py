from __future__ import annotations

from typing import Dict

import torch

from ..config import CompressionMode, PipelineConfig
from ..types import BranchArtifact, PipelineState, iter_named_parameters
from .base import CompressionBranch


class SparsityBranch(CompressionBranch):
    name = "sparsity"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self.branch_config = config.branches.sparsity

    @property
    def mode(self) -> CompressionMode:
        return CompressionMode.HYBRID

    def run(self, state: PipelineState) -> BranchArtifact:
        student = state.bundle.student
        target_sparsity = self.branch_config.target_sparsity

        pruned: Dict[str, torch.Tensor] = {}
        total = 0
        nonzero = 0
        effective_params: Dict[str, int] = {}

        sample_limit = 1_000_000

        for name, weight in iter_named_parameters(student):
            abs_weight = weight.abs()
            magnitude = abs_weight.view(-1)
            if magnitude.numel() == 0:
                continue

            float_magnitude = magnitude.to(torch.float32)
            if float_magnitude.numel() > sample_limit:
                indices = torch.randint(0, float_magnitude.numel(), (sample_limit,), device=float_magnitude.device)
                sample = float_magnitude[indices].cpu()
            else:
                sample = float_magnitude.cpu()

            threshold = torch.quantile(sample, target_sparsity)
            threshold_cast = threshold.to(abs_weight.device, dtype=abs_weight.dtype)
            mask = (abs_weight >= threshold_cast).to(weight.dtype)
            pruned_weight = weight * mask

            total += weight.numel()
            nonzero += mask.count_nonzero().item()
            pruned[name] = pruned_weight
            effective_params[name] = mask.count_nonzero().item()

        sparsity_level = 1.0 - (nonzero / max(total, 1))
        metrics = {
            "achieved_sparsity": sparsity_level,
            "target_sparsity": target_sparsity,
        }

        metadata = {
            "bregman_iterations": self.branch_config.bregman_iterations,
            "outlier_weighting": self.branch_config.enable_outlier_weighting,
            "effective_params": effective_params,
        }

        return BranchArtifact(name=self.name, state_dict=pruned, metadata=metadata, metrics=metrics)


__all__ = ["SparsityBranch"]


