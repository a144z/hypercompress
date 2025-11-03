from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


def _get_parent_and_attr(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_embedding_low_rank(model: nn.Module, rank: int) -> int:
    """Replace Embedding(V,D) with Embedding(V,r) + Linear(r,D) using SVD init.

    Returns number of embeddings replaced.
    """
    replaced = 0
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Embedding):
            continue

        vocab, dim = module.weight.shape
        target_r = max(1, min(rank, dim - 1))
        if target_r >= dim:
            continue

        w = module.weight.detach().to(torch.float32)
        try:
            u, s, vh = torch.linalg.svd(w, full_matrices=False)
        except RuntimeError:
            u, s, vh = torch.linalg.svd(w.cpu(), full_matrices=False)
            u = u.to(w.device)
            s = s.to(w.device)
            vh = vh.to(w.device)

        u_r = u[:, :target_r]  # (V, r)
        s_r = s[:target_r]
        vh_r = vh[:target_r, :]  # (r, D)

        emb_low = nn.Embedding(vocab, target_r)
        proj = nn.Linear(target_r, dim, bias=False)

        emb_low.weight.data.copy_(u_r.to(emb_low.weight.dtype))
        # For projection: W = S @ Vh, but Linear weight is W.T
        proj_weight = (vh_r.T * s_r).to(proj.weight.dtype)
        proj.weight.data.copy_(proj_weight)

        seq = nn.Sequential(emb_low, proj).to(device=device, dtype=module.weight.dtype)

        parent, attr = _get_parent_and_attr(model, name)
        setattr(parent, attr, seq)
        replaced += 1

    return replaced


__all__ = ["apply_embedding_low_rank"]


