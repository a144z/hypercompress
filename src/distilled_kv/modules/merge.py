from __future__ import annotations

from typing import Dict, Iterable

import torch

from ..logging import get_logger
from ..types import BranchArtifact


class MergeStrategy:
    """Combine branch artifacts using spectral weighting."""

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def merge(self, base_state: Dict[str, torch.Tensor], artifacts: Iterable[BranchArtifact]) -> Dict[str, torch.Tensor]:
        merged = {name: tensor.clone() for name, tensor in base_state.items()}

        for artifact in artifacts:
            weight = self._branch_weight(artifact)
            self.logger.info("Merging branch %s with weight %.4f", artifact.name, weight)

            applied_any = False
            for name, tensor in artifact.state_dict.items():
                if name not in merged:
                    # Parameter doesn't exist in student - skip it
                    # (Branches operate on teacher, but we only merge into student)
                    self.logger.debug(
                        "Skipping parameter %s from branch %s (not in student model)",
                        name,
                        artifact.name,
                    )
                    continue

                base_tensor = merged[name]
                
                # Check if shapes are compatible for merging
                if base_tensor.shape == tensor.shape:
                    # Same shape - blend them
                    merged[name] = (1 - weight) * base_tensor + weight * tensor
                    self.logger.debug("Blended parameter %s (shape %s)", name, base_tensor.shape)
                    applied_any = True
                elif self._can_reshape(base_tensor, tensor):
                    # Can be reshaped to match - try to reshape and merge
                    try:
                        reshaped = tensor.reshape(base_tensor.shape)
                        merged[name] = (1 - weight) * base_tensor + weight * reshaped
                        self.logger.info(
                            "Reshaped and merged parameter %s (%s -> %s) from branch %s",
                            name,
                            tensor.shape,
                            base_tensor.shape,
                            artifact.name,
                        )
                        applied_any = True
                    except RuntimeError:
                        # Reshape failed - skip this parameter
                        self.logger.warning(
                            "Cannot reshape %s from %s to %s, skipping branch %s update",
                            name,
                            tensor.shape,
                            base_tensor.shape,
                            artifact.name,
                        )
                else:
                    # Different shape that can't be reshaped - skip to avoid errors
                    # This can happen when branches compress dimensions in ways incompatible with student
                    self.logger.warning(
                        "Skipping incompatible parameter %s (branch shape %s vs student shape %s) from branch %s",
                        name,
                        tensor.shape,
                        base_tensor.shape,
                        artifact.name,
                    )

            artifact.metadata["applied"] = applied_any
            if not applied_any:
                artifact.metrics = {
                    key: (1.0 if isinstance(value, (int, float)) else value)
                    for key, value in artifact.metrics.items()
                }

        return merged

    def _branch_weight(self, artifact: BranchArtifact) -> float:
        """Compute merge weight for a branch artifact.
        
        For structural transformations, use full weight (1.0) since they replace layers.
        For effective-only compressions, use weighted blend based on compression ratio.
        """
        # Structural transformations should fully replace, not blend
        if artifact.metadata.get("structural_mode", False) or artifact.metadata.get("structurally_replaced", 0) > 0:
            return 1.0
        
        compression = artifact.metrics.get("compression_ratio") or artifact.metrics.get("embedding_reduction") or artifact.metrics.get("cache_reduction")
        if compression and compression > 1.0:
            # Higher compression -> more aggressive replacement
            return min(1.0, max(0.7, 1.0 / (compression ** 0.5)))
        return 0.8  # Default to more aggressive blending for hypercompression

    def _can_reshape(self, target: torch.Tensor, source: torch.Tensor) -> bool:
        """Check if source tensor can be reshaped to target shape."""
        if target.numel() != source.numel():
            return False
        return True


__all__ = ["MergeStrategy"]


