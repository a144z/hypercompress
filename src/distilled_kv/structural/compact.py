from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


@dataclass
class StructuralLRAReport:
    replaced: int
    original_params: Dict[str, int]
    new_params: Dict[str, int]

    @property
    def total_original(self) -> int:
        return sum(self.original_params.values())

    @property
    def total_new(self) -> int:
        return sum(self.new_params.values())

    @property
    def ratio(self) -> float:
        total_new = self.total_new or 1
        return self.total_original / total_new if self.total_original else 1.0


def _replace_module(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    setattr(parent, name, new_module)


def _get_parent_and_attr(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_structural_lra(
    model: nn.Module,
    rank: int,
    max_rank_fraction: float = 0.05,
) -> StructuralLRAReport:
    """Replace Linear layers with low-rank factorization (two Linear layers)."""

    replaced = 0
    original_params: Dict[str, int] = {}
    new_params: Dict[str, int] = {}
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        weight = module.weight.data
        out_dim, in_dim = weight.shape
        max_r = max(1, int(max_rank_fraction * min(out_dim, in_dim)))
        target_r = min(rank, max_r)
        if target_r >= min(out_dim, in_dim):
            continue

        # Track original parameter names before replacement
        for param_name, param in module.named_parameters():
            original_params[f"{name}.{param_name}"] = param.numel()

        w = weight.detach().to(torch.float32)
        try:
            u, s, v = torch.linalg.svd(w, full_matrices=False)
        except RuntimeError:
            # Fallback CPU SVD
            u, s, v = torch.linalg.svd(w.cpu(), full_matrices=False)
            u = u.to(w.device)
            s = s.to(w.device)
            v = v.to(w.device)

        u_r = (u[:, :target_r] * s[:target_r].unsqueeze(0))  # (out, r)
        v_r = v[:target_r, :]  # (r, in)

        first = nn.Linear(in_dim, target_r, bias=False)
        second = nn.Linear(target_r, out_dim, bias=module.bias is not None)

        first.weight.data.copy_(v_r.to(first.weight.dtype))
        second.weight.data.copy_(u_r.to(second.weight.dtype))
        if module.bias is not None:
            second.bias.data.copy_(module.bias.data)

        seq = nn.Sequential(first, second).to(device=device, dtype=module.weight.dtype)

        parent, attr = _get_parent_and_attr(model, name)
        _replace_module(parent, attr, seq)
        replaced += 1

        for sub_name, param in seq.named_parameters():
            new_params[f"{name}.{sub_name}"] = param.numel()

    return StructuralLRAReport(replaced=replaced, original_params=original_params, new_params=new_params)


__all__ = ["StructuralLRAReport", "apply_structural_lra"]


