from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from distilled_kv import CompressionPipeline, ModelBundle
from distilled_kv.config import PipelineConfig
from distilled_kv.utils.models import TinyByteLM


class DummyDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int):
        tokens = torch.randint(0, 64, (16,))
        return {"input_ids": tokens, "labels": tokens.clone()}


def test_pipeline_runs(tmp_path):
    config = PipelineConfig()
    config.logging.log_dir = str(tmp_path / "logs")
    config.storage.checkpoint_dir = str(tmp_path / "ckpt")
    config.logging.tensorboard = False

    teacher = TinyByteLM(vocab_size=64, hidden_size=64)
    student = TinyByteLM(vocab_size=64, hidden_size=32)

    bundle = ModelBundle(teacher=teacher, student=student, label="test")
    pipeline = CompressionPipeline(config, bundle, work_dir=tmp_path)

    loader = DataLoader(DummyDataset(), batch_size=2)
    artifacts = pipeline.run(distill_loader=loader, finetune_loader=loader)

    assert artifacts.checkpoint.exists()
    assert artifacts.state.merged_state is not None
    assert artifacts.summary.effective_ratio >= 1.0
    assert artifacts.summary.target_ratio >= 1.0

