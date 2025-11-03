from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

from ..config import PipelineConfig
from ..structural import StructuralLRAReport
from ..types import BranchArtifact, PipelineState
from ..utils.model_stats import count_parameters


@dataclass
class CompressionSummary:
    effective_ratio: float
    target_ratio: float
    meets_target: bool
    base_params: int
    student_params: int
    branch_ratios: Dict[str, float]
    branch_metrics: Dict[str, Dict[str, float]]
    notes: List[str]

    def to_metrics(self) -> Dict[str, float]:
        metrics = {
            "compression/effective_ratio": self.effective_ratio,
            "compression/target_ratio": self.target_ratio,
            "compression/base_params": float(self.base_params),
            "compression/student_params": float(self.student_params),
        }
        for name, ratio in self.branch_ratios.items():
            metrics[f"compression/{name}_ratio"] = ratio
        return metrics

    def as_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["metrics"] = self.to_metrics()
        return result


def build_compression_summary(config: PipelineConfig, state: PipelineState) -> CompressionSummary:
    target_ratio = config.targets.compression_ratio

    branch_ratios: Dict[str, float] = {}
    branch_metrics: Dict[str, Dict[str, float]] = {}

    baseline_params_map = state.distillation_history.get("baseline_student_params", {})

    # Count parameters AFTER structural transformations (structural LRA/BLT run before summary)
    student_params_map = {
        name: param.numel() for name, param in state.bundle.student.named_parameters()
    }
    effective_counts = dict(student_params_map)

    for name, artifact in state.branch_results.items():
        branch_metrics[name] = artifact.metrics
        branch_ratios[name] = _estimate_branch_ratio(name, artifact, baseline_params_map or student_params_map)

        effective_meta = artifact.metadata.get("effective_params") if artifact.metadata else None
        if effective_meta:
            for param_name, count in effective_meta.items():
                if param_name not in effective_counts:
                    continue
                effective_counts[param_name] = min(effective_counts[param_name], int(count))

    base_params = count_parameters(state.bundle.teacher)
    # Re-count student params to reflect structural changes (embeddings, LRA layers replaced)
    student_params = count_parameters(state.bundle.student)

    effective_total = sum(effective_counts.values()) if effective_counts else student_params
    effective_total = max(effective_total, 1)
    effective_ratio = base_params / effective_total

    # Actual parameter ratio reflects structural changes (real parameter count reduction)
    param_ratio = base_params / max(student_params, 1)
    branch_ratios.setdefault("param_ratio", param_ratio)
    
    # Track structural transformation impact
    structural_report = state.distillation_history.get("structural_lra_report")
    if isinstance(structural_report, StructuralLRAReport) and structural_report.replaced > 0:
        branch_ratios["structural_lra"] = structural_report.ratio

    blt_artifact = state.branch_results.get("blt")
    if blt_artifact and "structural_ratio" in blt_artifact.metrics:
        branch_ratios["blt_structural"] = blt_artifact.metrics["structural_ratio"]

    meets_target = effective_ratio >= target_ratio

    notes: List[str] = []
    if not branch_ratios:
        notes.append("No branch ratios were produced; using defaults of 1.0")
    if not meets_target:
        notes.append(
            f"Effective ratio {effective_ratio:.2f}x below target {target_ratio:.2f}x. Consider enabling more branches or adjusting configuration."
        )

    return CompressionSummary(
        effective_ratio=effective_ratio,
        target_ratio=target_ratio,
        meets_target=meets_target,
        base_params=base_params,
        student_params=student_params,
        branch_ratios=branch_ratios,
        branch_metrics=branch_metrics,
        notes=notes,
    )


def _estimate_branch_ratio(name: str, artifact: BranchArtifact, param_counts: Dict[str, int]) -> float:
    metrics = artifact.metrics
    effective_meta = artifact.metadata.get("effective_params") if artifact.metadata else None

    if not metrics or not effective_meta:
        return metrics.get("compression_ratio", 1.0) if metrics else 1.0

    original = 0
    effective = 0
    for param_name, eff_count in effective_meta.items():
        original += param_counts.get(param_name, eff_count)
        effective += eff_count

    if original == 0 or effective == 0:
        return 1.0

    return max(original / effective, 1.0)


__all__ = ["CompressionSummary", "build_compression_summary"]


