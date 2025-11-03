from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from ..config import CompressionMode, PipelineConfig
from ..types import BranchArtifact, PipelineState, iter_named_parameters
from .base import CompressionBranch


def _get_parent_and_attr(model: nn.Module, module_name: str):
    """Helper to get parent module and attribute name."""
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


class BLTBranch(CompressionBranch):
    name = "blt"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self.branch_config = config.branches.blt
        # Enable structural transformations for high compression targets
        self.apply_structural = config.targets.compression_ratio > 10.0

    @property
    def mode(self) -> CompressionMode:
        return CompressionMode.BLT

    def run(self, state: PipelineState) -> BranchArtifact:
        student = state.bundle.student
        reduction = self.branch_config.embedding_reduction
        latent_dim = self.branch_config.latent_dim

        compressed: Dict[str, torch.Tensor] = {}
        new_param_total = 0
        original_total = 0
        effective_params: Dict[str, int] = {}
        structurally_replaced = 0
        structural_original_params: Dict[str, int] = {}
        structural_new_params: Dict[str, int] = {}
        removed_param_names: list[str] = []

        # For structural compression, ALWAYS apply it (even for tied weights)
        # We'll handle tied weights by compressing both input and output embeddings together
        if self.apply_structural and reduction > 0.1:
            self.logger.info("Applying structural embedding compression with latent_dim=%d", latent_dim)
            
            for name, module in list(student.named_modules()):
                if not isinstance(module, nn.Embedding):
                    continue
                
                vocab, dim = module.weight.shape
                target_r = max(1, min(int(latent_dim), int(dim * (1 - reduction))))
                if target_r >= dim:
                    self.logger.debug("Skipping %s: target_r=%d >= dim=%d", name, target_r, dim)
                    continue

                original_total += int(module.weight.numel())
                structural_original_params[f"{name}.weight"] = int(module.weight.numel())
                removed_param_names.append(f"{name}.weight")
                
                # Perform SVD for structural replacement
                w = module.weight.detach().to(torch.float32)
                try:
                    u, s, vh = torch.linalg.svd(w, full_matrices=False)
                except RuntimeError:
                    u, s, vh = torch.linalg.svd(w.cpu(), full_matrices=False)
                    u = u.to(w.device)
                    s = s.to(w.device)
                    vh = vh.to(w.device)

                u_r = u[:, :target_r]  # (V, r)
                s_r = s[:target_r]  # (r,)
                vh_r = vh[:target_r, :]  # (r, D)

                # Create structural replacement: Embedding(V,r) + Linear(r,D)
                device = next(student.parameters()).device if any(True for _ in student.parameters()) else torch.device("cpu")
                emb_low = nn.Embedding(vocab, target_r)
                proj = nn.Linear(target_r, dim, bias=False)

                # For embedding: use U (left singular vectors) directly
                emb_low.weight.data.copy_(u_r.to(emb_low.weight.dtype))
                # For projection weight: W = S @ Vh_r which has shape (r, D)
                # The nn.Linear weight needs to be (D, r), so we use the transpose
                proj_weight = (vh_r.T * s_r).to(proj.weight.dtype)
                proj.weight.data.copy_(proj_weight)

                seq = nn.Sequential(emb_low, proj).to(device=device, dtype=module.weight.dtype)
                parent, attr = _get_parent_and_attr(student, name)
                setattr(parent, attr, seq)

                structurally_replaced += 1

                module_new_total = 0
                for sub_name, param in seq.named_parameters():
                    full_name = f"{name}.{sub_name}"
                    count = int(param.numel())
                    structural_new_params[full_name] = count
                    effective_params[full_name] = count
                    module_new_total += count

                new_param_total += module_new_total
                self.logger.info("Structurally replaced embedding %s: (%d,%d) -> Embedding(%d,%d)+Linear(%d,%d)", 
                                name, vocab, dim, vocab, target_r, target_r, dim)
        else:
            # Traditional projection-based compression (for low compression ratios or when disabled)
            self.logger.info("Applying projection-based embedding compression")
            for name, weight in iter_named_parameters(student):
                if "embed" not in name.lower():
                    continue

                original_total += weight.numel()
                matrix = weight.to(torch.float32)
                new_dim = max(1, int(matrix.shape[-1] * (1 - reduction)))
                new_dim = min(new_dim, latent_dim)

                if new_dim >= matrix.shape[-1]:
                    continue

                projection = torch.randn(matrix.shape[-1], new_dim, device=matrix.device, dtype=matrix.dtype)
                projected = matrix @ projection
                approx = projected @ projection.t()
                compressed[name] = approx.to(weight.dtype)
                reduced_count = projected.numel() + projection.numel()
                new_param_total += reduced_count
                effective_params[name] = reduced_count
                self.logger.info("Projection-compressed %s: %s -> latent_dim=%d", name, weight.shape, new_dim)

        reduction_ratio = (original_total / max(new_param_total, 1)) if new_param_total else 1.0
        metrics = {
            "embedding_reduction": reduction_ratio,
            "latent_dim": float(latent_dim),
            "structural_replacements": float(structurally_replaced),
        }

        metadata = {
            "reduction_target": reduction,
            "multimodal": self.branch_config.multimodal,
            "effective_params": effective_params,
            "structural_mode": self.apply_structural,
            "structurally_replaced": structurally_replaced,
            "structural_original_params": structural_original_params,
            "structural_new_params": structural_new_params,
            "structural_removed_params": removed_param_names,
        }

        return BranchArtifact(name=self.name, state_dict=compressed, metadata=metadata, metrics=metrics)


__all__ = ["BLTBranch"]
