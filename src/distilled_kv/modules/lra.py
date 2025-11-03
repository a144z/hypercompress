from __future__ import annotations

from math import prod
from typing import Dict

import torch

from ..config import CompressionMode, PipelineConfig
from ..types import BranchArtifact, PipelineState, iter_named_parameters
from .base import CompressionBranch


class LRABranch(CompressionBranch):
    name = "lra"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self.branch_config = config.branches.lra
        self.threshold = config.targets.max_rank_fraction

    @property
    def mode(self) -> CompressionMode:
        return CompressionMode.LRA

    def run(self, state: PipelineState) -> BranchArtifact:  # noqa: D401
        student = state.bundle.student
        rank = self.branch_config.rank

        total_params = 0
        factored_params = 0
        applied_ranks: list[int] = []
        compressed: Dict[str, torch.Tensor] = {}
        effective_params: Dict[str, int] = {}

        for name, weight in iter_named_parameters(student):
            if weight.ndim < 2:
                continue

            rows, cols = weight.shape[0], prod(weight.shape[1:])
            matrix = weight.reshape(rows, cols).to(torch.float32)

            max_rank = max(1, int(self.threshold * min(rows, cols)))
            target_rank = min(rank, max_rank)

            if target_rank >= min(rows, cols):
                compressed[name] = weight.detach().clone()
                factored_params += weight.numel()
                total_params += weight.numel()
                applied_ranks.append(min(rows, cols))
                effective_params[name] = weight.numel()
                continue

            u, s, v = torch.linalg.svd(matrix, full_matrices=False)
            u_r = u[:, :target_rank]
            s_r = s[:target_rank]
            v_r = v[:target_rank, :]

            approx = (u_r * s_r.unsqueeze(0)) @ v_r
            approx = approx.reshape_as(weight).to(weight.dtype)
            compressed[name] = approx

            total_params += weight.numel()
            factored_params += u_r.numel() + s_r.numel() + v_r.numel()
            applied_ranks.append(target_rank)
            effective_params[name] = u_r.numel() + s_r.numel() + v_r.numel()

        compression_ratio = (total_params / max(factored_params, 1)) if factored_params else 1.0
        metrics = {
            "compression_ratio": compression_ratio,
            "mean_rank": float(sum(applied_ranks) / max(len(applied_ranks), 1)),
        }

        metadata = {"rank": rank, "effective_params": effective_params}

        return BranchArtifact(name=self.name, state_dict=compressed, metadata=metadata, metrics=metrics)


__all__ = ["LRABranch"]


