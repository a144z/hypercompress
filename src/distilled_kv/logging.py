from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console
from rich.logging import RichHandler

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore


def configure_logging(log_dir: str, experiment: str, level: int = logging.INFO) -> None:
    """Configure root logging with Rich console and file handlers."""

    log_directory = Path(log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / f"{experiment}.log"

    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            RichHandler(console=Console(stderr=True), show_time=False, rich_tracebacks=True),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a namespaced logger."""

    return logging.getLogger(name if name else "distilled_kv")


@dataclass
class LoggingManager(AbstractContextManager["LoggingManager"]):
    """Helper managing experiment logging backends."""

    experiment: str
    log_dir: Path
    enable_wandb: bool = False
    wandb_project: Optional[str] = None
    tensorboard: bool = True

    def __post_init__(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer: Optional[SummaryWriter] = None
        self._wandb_run = None

        configure_logging(str(self.log_dir), self.experiment)
        self.logger = get_logger(__name__)

        if self.tensorboard and SummaryWriter:
            tb_path = self.log_dir / "tensorboard"
            tb_path.mkdir(exist_ok=True, parents=True)
            self._writer = SummaryWriter(log_dir=str(tb_path))

        if self.enable_wandb and wandb:
            self._wandb_run = wandb.init(project=self.wandb_project, name=self.experiment, reinit=True)
        elif self.enable_wandb:
            self.logger.warning("W&B requested but not installed. Skipping remote logging.")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = "") -> None:
        if not metrics:
            return

        scoped = {f"{prefix}{key}" if prefix else key: value for key, value in metrics.items()}
        self.logger.info("Metrics %s", scoped)

        if self._writer:
            for key, value in scoped.items():
                self._writer.add_scalar(key, value, global_step=step)

        if self._wandb_run:
            self._wandb_run.log(scoped, step=step)

    def log_text(self, title: str, content: str) -> None:
        self.logger.info("%s\n%s", title, content)
        if self._wandb_run:
            self._wandb_run.log({title: wandb.Html(f"<pre>{content}</pre>")})  # type: ignore[attr-defined]

    def close(self) -> None:
        if self._writer:
            self._writer.flush()
            self._writer.close()
            self._writer = None

        if self._wandb_run:
            self._wandb_run.finish()
            self._wandb_run = None

    def __exit__(self, exc_type, exc, exc_tb) -> None:  # type: ignore[override]
        self.close()


__all__ = ["configure_logging", "get_logger", "LoggingManager"]


