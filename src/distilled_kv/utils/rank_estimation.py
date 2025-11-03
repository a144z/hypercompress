from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch

from ..logging import get_logger
from ..types import PipelineState, iter_named_parameters


@dataclass
class RankStatistics:
    layer_ranks: Dict[str, int]
    spectral_energy: Dict[str, float]

    def mean_rank(self) -> float:
        if not self.layer_ranks:
            return 0.0
        return sum(self.layer_ranks.values()) / len(self.layer_ranks)


def estimate_ranks(state: PipelineState) -> RankStatistics:
    logger = get_logger(__name__)
    ranks: Dict[str, int] = {}
    energies: Dict[str, float] = {}

    for name, tensor in iter_named_parameters(state.bundle.teacher):
        if tensor.ndim < 2:
            continue

        matrix = tensor.reshape(tensor.shape[0], -1).to(torch.float32)
        u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
        cumulative = torch.cumsum(s, dim=0)
        threshold = 0.9 * cumulative[-1]
        rank = int(torch.searchsorted(cumulative, threshold).item() + 1)

        ranks[name] = rank
        energies[name] = float(s[:rank].sum() / s.sum())
        logger.debug("Layer %s estimated rank %d", name, rank)

    return RankStatistics(layer_ranks=ranks, spectral_energy=energies)


def select_adaptive_rank(ranks: Iterable[int], max_fraction: float, dim: int) -> int:
    rank_list = list(ranks)
    if not rank_list:
        return max(1, int(max_fraction * dim))

    candidate = int(sum(rank_list) / len(rank_list))
    return min(candidate, max(1, int(max_fraction * dim)))


__all__ = ["RankStatistics", "estimate_ranks", "select_adaptive_rank"]


