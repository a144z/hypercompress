from __future__ import annotations

from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader
import typer
from transformers import AutoTokenizer

from distilled_kv import CompressionPipeline, CompressionSummary, ModelBundle, PipelineArtifacts, load_config
from distilled_kv.utils.data import TextTokenDataset
from distilled_kv.utils.hf_loader import HFLoaderConfig, load_causal_lm


app = typer.Typer(help="Run Distilled KV with Hugging Face Transformers")


def _build_loader(
    dataset_path: Optional[Path], tokenizer, seq_len: int, stride: int, batch_size: int, max_chunks: int
) -> Optional[DataLoader]:
    if dataset_path is None:
        return None

    resolved_path = Path(dataset_path)
    if not resolved_path.exists():
        raise FileNotFoundError(resolved_path)

    dataset = TextTokenDataset(
        file_path=resolved_path, tokenizer=tokenizer, seq_len=seq_len, stride=stride, max_chunks=max_chunks
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


@app.command()
def run(
    config: Path = typer.Option(..., exists=True, help="YAML pipeline configuration"),
    ratio: float = typer.Option(None, help="Target effective compression ratio override (e.g., 100.0)"),
    teacher_model: str = typer.Option(..., help="Hugging Face model name for the teacher (e.g. microsoft/phi-2, meta-llama/Llama-2-7b-hf)"),
    student_model: str = typer.Option(..., help="Hugging Face model name for the student"),
    tokenizer_model: Optional[str] = typer.Option(None, help="Tokenizer model (defaults to teacher)"),
    tokenizer_revision: Optional[str] = typer.Option(None, help="Tokenizer revision"),
    teacher_revision: Optional[str] = typer.Option(None, help="Teacher model revision"),
    student_revision: Optional[str] = typer.Option(None, help="Student model revision"),
    dtype: str = typer.Option("bfloat16", help="Torch dtype to load models with (e.g. float16, bfloat16)"),
    teacher_device_map: Optional[str] = typer.Option(None, help="Device map for teacher (e.g. auto, cuda:0)"),
    student_device_map: Optional[str] = typer.Option(None, help="Device map for student"),
    cache_dir: Optional[Path] = typer.Option(None, help="Hugging Face cache directory"),
    trust_remote_code: bool = typer.Option(False, help="Allow loading models with custom code"),
    distill_data: Optional[Path] = typer.Option(None, help="Path to text file for distillation dataset"),
    finetune_data: Optional[Path] = typer.Option(None, help="Path to text file for finetuning dataset"),
    seq_len: int = typer.Option(512, help="Sequence length for tokenized chunks (smaller = faster)"),
    stride: int = typer.Option(256, help="Stride for tokenized chunks"),
    batch_size: int = typer.Option(1, help="Batch size for DataLoaders"),
    max_chunks: int = typer.Option(100, help="Maximum chunks per dataset (limits data size for speed)"),
    load_in_8bit: bool = typer.Option(False, help="Load models in 8-bit quantization (requires bitsandbytes)"),
    load_in_4bit: bool = typer.Option(False, help="Load models in 4-bit quantization (requires bitsandbytes >= 0.41)"),
    work_dir: Optional[Path] = typer.Option(None, help="Working directory for artifacts"),
) -> None:
    """Launch the hybrid compression pipeline on large Hugging Face models."""

    if load_in_8bit and load_in_4bit:
        raise typer.BadParameter("Choose either 8-bit or 4-bit quantization, not both")

    overrides = {"targets": {"compression_ratio": ratio}} if ratio else None
    cfg = load_config(config, overrides=overrides)

    # Check for auth token from environment variables (optional - only needed for gated models)
    import os
    auth_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None

    teacher_cfg = HFLoaderConfig(
        model_name=teacher_model,
        revision=teacher_revision,
        dtype=dtype,
        device_map=teacher_device_map,
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=trust_remote_code,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        use_auth_token=auth_token,
    )
    typer.echo(f"Loading teacher model: {teacher_model}")
    try:
        teacher, tokenizer = load_causal_lm(teacher_cfg)
    except Exception as e:
        typer.echo(f"❌ Failed to load teacher model: {e}", err=True)
        typer.echo("💡 Tip: If this is a gated model, run: huggingface-cli login", err=True)
        typer.echo("💡 Or pass --hf-token YOUR_TOKEN", err=True)
        raise typer.Exit(1)

    if tokenizer_model:
        typer.echo(f"Loading tokenizer: {tokenizer_model}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_model,
                revision=tokenizer_revision,
                use_fast=True,
                trust_remote_code=trust_remote_code,
                cache_dir=str(cache_dir) if cache_dir else None,
                token=auth_token,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            typer.echo(f"⚠️  Failed to load separate tokenizer, using teacher tokenizer: {e}", err=True)

    student_cfg = HFLoaderConfig(
        model_name=student_model,
        revision=student_revision,
        dtype=dtype,
        device_map=student_device_map,
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=trust_remote_code,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        use_auth_token=auth_token,
    )
    typer.echo(f"Loading student model: {student_model}")
    try:
        student, _ = load_causal_lm(student_cfg)
    except Exception as e:
        typer.echo(f"❌ Failed to load student model: {e}", err=True)
        typer.echo("💡 Tip: If this is a gated model, run: huggingface-cli login", err=True)
        typer.echo("💡 Or pass --hf-token YOUR_TOKEN", err=True)
        raise typer.Exit(1)

    bundle = ModelBundle(teacher=teacher, student=student, tokenizer=tokenizer, label=cfg.logging.experiment)
    pipeline = CompressionPipeline(cfg, bundle, work_dir=work_dir)

    try:
        typer.echo(f"Loading distillation dataset (max {max_chunks} chunks)...")
        distill_loader = _build_loader(distill_data, tokenizer, seq_len, stride, batch_size, max_chunks)
        typer.echo(f"✓ Distillation dataset loaded: {len(distill_loader.dataset)} chunks")
    except FileNotFoundError as exc:
        typer.echo(f"❌ Distillation dataset not found: {exc}", err=True)
        typer.echo("💡 Create the file with plain text (one large corpus).", err=True)
        raise typer.Exit(1)

    effective_finetune = finetune_data or distill_data
    try:
        typer.echo(f"Loading finetune dataset (max {max_chunks} chunks)...")
        finetune_loader = _build_loader(effective_finetune, tokenizer, seq_len, stride, batch_size, max_chunks)
        typer.echo(f"✓ Finetune dataset loaded: {len(finetune_loader.dataset)} chunks")
    except FileNotFoundError as exc:
        typer.echo(f"❌ Finetune dataset not found: {exc}", err=True)
        typer.echo("💡 Provide --finetune-data or omit to reuse --distill-data.", err=True)
        raise typer.Exit(1)

    typer.echo("Starting compression pipeline... this may take some time for large models.")
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


