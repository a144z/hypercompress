from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import CompressionMode, PipelineConfig
from ..logging import get_logger
from ..types import BranchArtifact, PipelineState


class CompressionBranch(ABC):
    """Interface implemented by all compression branches."""

    name: str = "branch"

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger = get_logger(f"distilled_kv.{self.name}")

    def should_run(self, mode: CompressionMode) -> bool:
        return mode in {CompressionMode.HYBRID, self.mode}

    @property
    @abstractmethod
    def mode(self) -> CompressionMode:
        ...

    def prepare(self, state: PipelineState) -> None:
        """Optional pre-processing hook that can mutate the pipeline state."""

    @abstractmethod
    def run(self, state: PipelineState) -> BranchArtifact:
        """Execute the branch, producing a state dictionary delta and metrics."""

    def finalize(self, state: PipelineState, artifact: BranchArtifact) -> None:
        """Optional hook invoked after merging."""


class BranchRegistry:
    """Registry responsible for instantiating enabled branches."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._registry: list[type[CompressionBranch]] = []

    def register(self, branch_cls: type[CompressionBranch]) -> None:
        self._registry.append(branch_cls)

    def create(self, mode: CompressionMode) -> list[CompressionBranch]:
        branches: list[CompressionBranch] = []
        for branch_cls in self._registry:
            branch = branch_cls(self.config)
            if branch.should_run(mode):
                branches.append(branch)
        return branches


__all__ = ["CompressionBranch", "BranchRegistry"]


