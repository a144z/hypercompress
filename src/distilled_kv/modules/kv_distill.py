from __future__ import annotations

from typing import Dict

import torch

from ..config import CompressionMode, PipelineConfig
from ..types import BranchArtifact, PipelineState, iter_named_parameters
from .base import CompressionBranch


class KVBranch(CompressionBranch):
    name = "kv_distill"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self.branch_config = config.branches.kv

    @property
    def mode(self) -> CompressionMode:
        return CompressionMode.KV

    def run(self, state: PipelineState) -> BranchArtifact:
        student = state.bundle.student
        cache_tokens = self.branch_config.cache_tokens

        compressed: Dict[str, torch.Tensor] = {}
        total_cache = 0
        distilled_cache = 0
        effective_params: Dict[str, int] = {}

        # Priority: Look for explicit KV projection layers (Transformer models)
        # Fallback: For non-Transformer models, compress any linear layer weights
        # (excluding embeddings which BLT handles)
        kv_patterns = ("k_proj", "v_proj", "attn", "query", "key", "value")
        
        for name, weight in iter_named_parameters(student):
            # Skip embeddings (handled by BLT branch) and 1D parameters
            if weight.ndim < 2:
                continue
            if "embed" in name.lower():
                continue

            # Check if this is a KV-related layer or a linear/transform layer
            is_kv = any(token in name.lower() for token in kv_patterns)
            # For fallback: match head/proj, or any weight in transform/linear structures
            is_fallback = (
                any(token in name.lower() for token in ("head", "proj"))
                or ("transform" in name.lower() and "weight" in name.lower())
                or ("linear" in name.lower() and "weight" in name.lower())
            )
            
            if not (is_kv or is_fallback):
                continue

            matrix = weight.reshape(weight.shape[0], -1).to(torch.float32)
            rows, cols = matrix.shape
            target_rank = min(cache_tokens, min(rows, cols))

            if target_rank == 0 or target_rank >= min(rows, cols):
                continue

            u, s, v = torch.linalg.svd(matrix, full_matrices=False)
            latent = (u[:, :target_rank] * s[:target_rank].unsqueeze(0)) @ v[:target_rank, :]
            latent = latent.reshape_as(weight).to(weight.dtype)

            compressed[name] = latent
            original_cache = rows * cols
            latent_cache = rows * target_rank + cols * target_rank

            total_cache += original_cache
            distilled_cache += latent_cache
            effective_params[name] = latent_cache
            
            self.logger.debug(
                "KV-compressed %s: shape %s -> rank %d (original %d params, compressed %d params)",
                name,
                weight.shape,
                target_rank,
                original_cache,
                latent_cache,
            )

        metrics = {
            "cache_reduction": (total_cache / max(distilled_cache, 1)) if distilled_cache else 1.0,
            "latent_tokens": float(cache_tokens),
        }

        metadata = {
            "cache_tokens": cache_tokens,
            "student_hidden": self.branch_config.student_hidden,
            "effective_params": effective_params,
        }

        return BranchArtifact(
            name=self.name,
            state_dict=compressed,
            metadata=metadata,
            metrics=metrics,
        )


__all__ = ["KVBranch"]


