from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
import typer

from distilled_kv import (
    CompressionPipeline,
    CompressionSummary,
    ModelBundle,
    PipelineArtifacts,
    load_config,
)
from distilled_kv.utils.models import TinyByteLM

app = typer.Typer(help="Run the Distilled KV compression pipeline")


class RandomLanguageDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, samples: int) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.samples = samples

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int):  # noqa: D401
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def build_random_loader(vocab_size: int, seq_len: int, samples: int, batch_size: int) -> DataLoader:
    dataset = RandomLanguageDataset(vocab_size, seq_len, samples)
    return DataLoader(dataset, batch_size=batch_size)


@app.command()
def run(
    config: Path = typer.Option(..., exists=True, help="Path to YAML configuration"),
    ratio: float = typer.Option(None, help="Target effective compression ratio override (e.g., 100.0)"),
    vocab_size: int = typer.Option(256, help="Vocabulary size for synthetic data"),
    seq_len: int = typer.Option(64, help="Sequence length for synthetic data"),
    samples: int = typer.Option(128, help="Number of synthetic samples for distillation"),
    batch_size: int = typer.Option(8, help="Batch size for synthetic loaders"),
) -> None:
    overrides = {"targets": {"compression_ratio": ratio}} if ratio else None
    cfg = load_config(config, overrides=overrides)

    teacher = TinyByteLM(vocab_size=vocab_size, hidden_size=256)
    student = TinyByteLM(vocab_size=vocab_size, hidden_size=128)

    bundle = ModelBundle(teacher=teacher, student=student, tokenizer=None, label=cfg.logging.experiment)
    pipeline = CompressionPipeline(cfg, bundle)

    distill_loader = build_random_loader(vocab_size, seq_len, samples, batch_size)
    finetune_loader = build_random_loader(vocab_size, seq_len, samples // 2, batch_size)

    artifacts: PipelineArtifacts = pipeline.run(
        distill_loader=distill_loader,
        finetune_loader=finetune_loader,
    )

    typer.echo(f"Compression complete. Checkpoint stored at {artifacts.checkpoint}")
    _print_summary(artifacts.summary)


def _print_summary(summary: CompressionSummary) -> None:
    typer.echo(
        f"Estimated effective compression ratio: {summary.effective_ratio:.2f}x (target {summary.target_ratio:.2f}x)"
    )
    typer.echo(
        f"Teacher params: {summary.base_params:,} | Student params: {summary.student_params:,}"
    )
    for name, ratio in summary.branch_ratios.items():
        typer.echo(f"  - {name}: {ratio:.2f}x")
    if summary.notes:
        typer.echo("Notes:")
        for note in summary.notes:
            typer.echo(f"  * {note}")


if __name__ == "__main__":
    app()


