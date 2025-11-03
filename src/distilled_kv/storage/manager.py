from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..config import StorageConfig
from ..logging import get_logger
from ..types import PipelineState
from ..utils.checkpoints import save_state_dict


class ArtifactManager:
    def __init__(self, config: StorageConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def persist(self, state: PipelineState) -> Path:
        if state.merged_state is None:
            raise RuntimeError("No merged state present for persistence")

        label = state.bundle.label or "compressed"
        output_path = self.checkpoint_dir / f"{label}.{self.config.export_format}"
        save_state_dict(state.merged_state, output_path)

        if self.config.save_pretrained and hasattr(state.bundle.student, "save_pretrained"):
            export_dir = self.checkpoint_dir / label
            export_dir.mkdir(parents=True, exist_ok=True)

            save_kwargs: Dict[str, object] = {"safe_serialization": self.config.hf_safe_serialization}

            self.logger.info("Exporting student model via save_pretrained to %s", export_dir)
            # Move to cpu to avoid device-specific serialization issues
            original_device = next(state.bundle.student.parameters()).device
            state.bundle.student.to("cpu")
            state.bundle.student.save_pretrained(export_dir, **save_kwargs)

            if state.bundle.tokenizer and hasattr(state.bundle.tokenizer, "save_pretrained"):
                state.bundle.tokenizer.save_pretrained(export_dir)

            # Return student to original device (best-effort)
            try:
                state.bundle.student.to(original_device)
            except Exception:  # pragma: no cover - device errors
                self.logger.warning("Failed to move model back to %s after saving", original_device)

        if self.config.push_to_hub and self.config.hub_repo:
            self._push_to_hub(output_path)

        return output_path

    def _push_to_hub(self, path: Path) -> None:  # pragma: no cover - network side effect
        try:
            from huggingface_hub import HfApi
        except ImportError:
            self.logger.warning("huggingface_hub not installed; skipping upload of %s", path)
            return

        api = HfApi()
        self.logger.info("Uploading %s to HF repo %s", path, self.config.hub_repo)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=self.config.hub_repo,
            repo_type="model",
        )


__all__ = ["ArtifactManager"]


