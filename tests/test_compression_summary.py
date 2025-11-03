from __future__ import annotations

from types import SimpleNamespace

import torch

from distilled_kv.analysis.report import CompressionSummary, build_compression_summary
from distilled_kv.config import PipelineConfig
from distilled_kv.structural import StructuralLRAReport
from distilled_kv.types import BranchArtifact, ModelBundle, PipelineState
from distilled_kv.utils.models import TinyByteLM


def test_build_compression_summary(tmp_path):
    config = PipelineConfig()
    teacher = TinyByteLM(vocab_size=32, hidden_size=64)
    student = TinyByteLM(vocab_size=32, hidden_size=16)

    bundle = ModelBundle(teacher=teacher, student=student, label="summary-test")
    state = PipelineState(bundle=bundle, work_dir=tmp_path, device=torch.device("cpu"))

    baseline_counts = {name: param.numel() for name, param in student.named_parameters()}
    state.distillation_history["baseline_student_params"] = baseline_counts
    state.distillation_history["baseline_student_total"] = sum(baseline_counts.values())

    effective_params = {name: max(1, count // 10) for name, count in baseline_counts.items()}

    artifact = BranchArtifact(
        name="lra",
        state_dict={},
        metrics={"compression_ratio": 50.0},
        metadata={"effective_params": effective_params},
    )
    state.branch_results["lra"] = artifact

    summary = build_compression_summary(config, state)

    assert isinstance(summary, CompressionSummary)
    assert summary.effective_ratio > summary.branch_ratios["param_ratio"]
    assert "lra" in summary.branch_ratios


def test_summary_tracks_structural_lra(tmp_path):
    config = PipelineConfig()
    teacher = TinyByteLM(vocab_size=16, hidden_size=32)
    student = TinyByteLM(vocab_size=16, hidden_size=32)

    baseline_counts = {name: param.numel() for name, param in student.named_parameters()}

    bundle = ModelBundle(teacher=teacher, student=student, label="structural-test")
    state = PipelineState(bundle=bundle, work_dir=tmp_path, device=torch.device("cpu"))
    state.distillation_history["baseline_student_params"] = baseline_counts
    state.distillation_history["baseline_student_total"] = sum(baseline_counts.values())

    linear = student.transform[1]
    rank = 4
    first = torch.nn.Linear(linear.in_features, rank, bias=False)
    second = torch.nn.Linear(rank, linear.out_features, bias=True)
    student.transform[1] = torch.nn.Sequential(first, second)

    report = StructuralLRAReport(
        replaced=1,
        original_params={
            "transform.1.weight": int(linear.weight.numel()),
            "transform.1.bias": int(linear.bias.numel()),
        },
        new_params={
            "transform.1.0.weight": int(first.weight.numel()),
            "transform.1.1.weight": int(second.weight.numel()),
            "transform.1.1.bias": int(second.bias.numel()),
        },
    )

    state.distillation_history["structural_lra_report"] = report
    state.branch_results["lra"] = BranchArtifact(name="lra", state_dict={}, metadata={}, metrics={})

    summary = build_compression_summary(config, state)

    assert summary.branch_ratios["structural_lra"] == report.ratio

