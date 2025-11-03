from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    parameters: Iterable[torch.nn.Parameter]
    if trainable_only:
        parameters = (param for param in model.parameters() if param.requires_grad)
    else:
        parameters = model.parameters()

    return int(sum(param.numel() for param in parameters))


__all__ = ["count_parameters"]


