from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


class CompressionMode(str, Enum):
    HYBRID = "hybrid"
    LRA = "lra"
    KV = "kv"
    BLT = "blt"


class TargetConfig(BaseModel):
    compression_ratio: float = Field(default=1000.0, ge=1.0)
    retained_accuracy: float = Field(default=0.95, ge=0.0, le=1.0)
    max_rank_fraction: float = Field(default=0.02, gt=0.0, le=1.0)


class LRAConfig(BaseModel):
    rank: int = Field(default=64, ge=1)
    sparsity: float = Field(default=0.99, ge=0.0, lt=1.0)
    enable_tensor_lra: bool = Field(default=True)
    expert_limit: Optional[int] = Field(default=None, ge=1)
    enable_structural: bool = Field(default=False)


class KVConfig(BaseModel):
    cache_tokens: int = Field(default=128, ge=1)
    student_hidden: int = Field(default=1024, ge=1)
    activation_matching_weight: float = Field(default=0.2, ge=0.0)


class BLTConfig(BaseModel):
    embedding_reduction: float = Field(default=0.9, ge=0.0, lt=1.0)
    latent_dim: int = Field(default=256, ge=1)
    multimodal: bool = Field(default=True)


class SparsityConfig(BaseModel):
    target_sparsity: float = Field(default=0.99, ge=0.0, lt=1.0)
    enable_outlier_weighting: bool = Field(default=True)
    bregman_iterations: int = Field(default=5, ge=1)


class LoggingConfig(BaseModel):
    experiment: str = "hypercompress-run"
    log_dir: str = ".artifacts/logs"
    enable_wandb: bool = False
    wandb_project: Optional[str] = None
    tensorboard: bool = True

    @model_validator(mode="after")
    def check_wandb(self) -> "LoggingConfig":
        if self.enable_wandb and not self.wandb_project:
            raise ValueError("wandb_project must be provided when enable_wandb is true")
        return self


class DistillationConfig(BaseModel):
    kl_weight: float = Field(default=1.0, ge=0.0)
    activation_weight: float = Field(default=1.0, ge=0.0)
    token_budget: int = Field(default=13_000_000, ge=1)
    hierarchical_lora: bool = Field(default=True)


class FinetuneConfig(BaseModel):
    max_tokens: int = Field(default=10_000_000, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    patience: int = Field(default=3, ge=1)
    ppl_tolerance: float = Field(default=0.05, ge=0.0)
    mmlu_target: float = Field(default=0.95, ge=0.0, le=1.0)


class EvaluationConfig(BaseModel):
    run_ppl: bool = True
    run_mmlu: bool = True
    run_gsm8k: bool = False
    run_glue: bool = False
    ablation_branches: bool = True


class StorageConfig(BaseModel):
    checkpoint_dir: str = ".artifacts/checkpoints"
    export_format: str = "safetensors"
    push_to_hub: bool = False
    hub_repo: Optional[str] = None
    save_pretrained: bool = False
    hf_safe_serialization: bool = True

    @model_validator(mode="after")
    def check_hub(self) -> "StorageConfig":
        if self.push_to_hub and not self.hub_repo:
            raise ValueError("hub_repo required when push_to_hub is enabled")
        return self


class BranchConfig(BaseModel):
    lra: LRAConfig = LRAConfig()
    kv: KVConfig = KVConfig()
    blt: BLTConfig = BLTConfig()
    sparsity: SparsityConfig = SparsityConfig()


class PipelineConfig(BaseModel):
    mode: CompressionMode = CompressionMode.HYBRID
    targets: TargetConfig = TargetConfig()
    branches: BranchConfig = BranchConfig()
    logging: LoggingConfig = LoggingConfig()
    distillation: DistillationConfig = DistillationConfig()
    finetune: FinetuneConfig = FinetuneConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    storage: StorageConfig = StorageConfig()


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> PipelineConfig:
    """Load a pipeline configuration from YAML with optional overrides."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if overrides:
        raw = _deep_update(raw, overrides)

    try:
        return PipelineConfig(**raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


__all__ = [
    "CompressionMode",
    "PipelineConfig",
    "TargetConfig",
    "BranchConfig",
    "load_config",
]

