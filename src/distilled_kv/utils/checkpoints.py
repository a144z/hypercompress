from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from ..logging import get_logger


def save_state_dict(state_dict: Dict[str, torch.Tensor], path: Path) -> None:
    logger = get_logger(__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
    logger.info("Saved checkpoint to %s", path)


def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


__all__ = ["save_state_dict", "load_state_dict"]


