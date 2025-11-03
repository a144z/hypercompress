from __future__ import annotations

from distilled_kv.config import PipelineConfig, load_config


def test_load_config(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("mode: hybrid\nlogging:\n  experiment: test\n")

    config = load_config(path)

    assert isinstance(config, PipelineConfig)
    assert config.logging.experiment == "test"

