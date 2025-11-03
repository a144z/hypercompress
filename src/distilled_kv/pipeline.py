from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from .analysis.report import CompressionSummary, build_compression_summary
from .config import PipelineConfig
from .distillation.distiller import KnowledgeDistiller
from .evaluation.suite import EvaluationInputs, EvaluationSuite
from .finetune.loop import FineTuner
from .logging import LoggingManager, get_logger
from .modules.base import BranchRegistry
from .modules.blt import BLTBranch
from .modules.kv_distill import KVBranch
from .modules.lra import LRABranch
from .modules.merge import MergeStrategy
from .modules.sparsity import SparsityBranch
from .storage.manager import ArtifactManager
from .types import EvalResult, ModelBundle, PipelineState, iter_named_parameters
from .utils.rank_estimation import estimate_ranks
from .structural.compact import StructuralLRAReport, apply_structural_lra
from .structural.embedding import apply_embedding_low_rank
from .utils.model_stats import count_parameters


@dataclass
class PipelineArtifacts:
    checkpoint: Path
    evaluation: EvalResult
    summary: CompressionSummary
    state: PipelineState


class CompressionPipeline:
    """High-level orchestrator implementing the hybrid compression flow."""

    def __init__(
        self,
        config: PipelineConfig,
        bundle: ModelBundle,
        work_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.bundle = bundle
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.work_dir = work_dir or Path(".artifacts") / datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)

        self.bundle.teacher.to(self.device)
        self.bundle.student.to(self.device)

        self.registry = BranchRegistry(config)
        self.registry.register(LRABranch)
        self.registry.register(KVBranch)
        self.registry.register(BLTBranch)
        self.registry.register(SparsityBranch)

        self.merger = MergeStrategy()
        self.distiller = KnowledgeDistiller(config.distillation)
        self.finetuner = FineTuner(config.finetune)
        self.evaluator = EvaluationSuite(config.evaluation)
        self.storage = ArtifactManager(config.storage)

    def run(
        self,
        distill_loader: Optional[torch.utils.data.DataLoader] = None,
        finetune_loader: Optional[torch.utils.data.DataLoader] = None,
        evaluation_inputs: Optional[EvaluationInputs] = None,
    ) -> PipelineArtifacts:
        logging_manager = LoggingManager(
            experiment=self.config.logging.experiment,
            log_dir=Path(self.config.logging.log_dir),
            enable_wandb=self.config.logging.enable_wandb,
            wandb_project=self.config.logging.wandb_project,
            tensorboard=self.config.logging.tensorboard,
        )

        progress_columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ]
        use_progress = HAS_RICH and not self.config.logging.enable_wandb

        with logging_manager:
            state = self._initial_state()
            
            if use_progress:
                with Progress(*progress_columns) as progress:
                    task1 = progress.add_task("🔍 Baseline analysis...", total=None)
                    self._prepare(state)
                    progress.update(task1, completed=True)
                    
                    task2 = progress.add_task("⚡ Running compression branches...", total=None)
                    self._run_branches(state, logging_manager)
                    self._handle_structural_blt(state)
                    progress.update(task2, completed=True)
                    
                    task3 = progress.add_task("🔀 Merging branches...", total=None)
                    self._merge(state)
                    progress.update(task3, completed=True)
                    
                    if distill_loader:
                        task4 = progress.add_task("🎓 Distilling knowledge...", total=len(distill_loader))
                        self._distill(state, distill_loader, progress=progress, task=task4 if use_progress else None)
                        progress.update(task4, completed=True)
                    
                    if finetune_loader:
                        task5 = progress.add_task("🔧 Fine-tuning...", total=len(finetune_loader))
                        self._finetune(state, finetune_loader, progress=progress, task=task5 if use_progress else None)
                        progress.update(task5, completed=True)
                    
                    task6 = progress.add_task("📊 Evaluating...", total=None)
                    evaluation = self._evaluate(state, evaluation_inputs)
                    state.evaluation = evaluation.metrics
                    progress.update(task6, completed=True)
                    
                    # Apply structural transformations BEFORE merge to actually reduce parameter count
                    task7 = progress.add_task("🧱 Structural LRA compaction...", total=None)
                    if self.config.branches.lra.enable_structural:
                        self.logger.info("Applying structural LRA with rank=%d", self.config.branches.lra.rank)
                        report = apply_structural_lra(
                            state.bundle.student,
                            rank=self.config.branches.lra.rank,
                            max_rank_fraction=self.config.targets.max_rank_fraction,
                        )
                        self._handle_structural_lra(state, report)
                    else:
                        self.logger.warning("Structural LRA is disabled - param_ratio will be low")
                    progress.update(task7, completed=True)

                    # BLT structural compression was already applied during branch execution
                    task8 = progress.add_task("✓ Embeddings compressed by BLT", total=None)
                    blt_structural = state.branch_results.get("blt")
                    if blt_structural and blt_structural.metadata.get("structurally_replaced", 0) > 0:
                        self.logger.info("✓ BLT structurally compressed %d embeddings", 
                                       blt_structural.metadata.get("structurally_replaced"))
                    progress.update(task8, completed=True)

                    state.merged_state = state.bundle.student.state_dict()

                    task9 = progress.add_task("💾 Saving checkpoint...", total=None)
                    summary = build_compression_summary(self.config, state)
                    logging_manager.log_metrics(summary.to_metrics())
                    if summary.notes:
                        for note in summary.notes:
                            self.logger.warning(note)
                    checkpoint = self.storage.persist(state)
                    progress.update(task9, completed=True)
            else:
                # Non-progress path (when rich not available or wandb enabled)
                state = self._initial_state()
                self._prepare(state)
                self._run_branches(state, logging_manager)
                self._handle_structural_blt(state)
                self._merge(state)

                if distill_loader:
                    self._distill(state, distill_loader)

                if finetune_loader:
                    self._finetune(state, finetune_loader)

                # Apply structural transformations
                if self.config.branches.lra.enable_structural:
                    self.logger.info("Applying structural LRA with rank=%d", self.config.branches.lra.rank)
                    report = apply_structural_lra(
                        state.bundle.student,
                        rank=self.config.branches.lra.rank,
                        max_rank_fraction=self.config.targets.max_rank_fraction,
                    )
                    self._handle_structural_lra(state, report)
                else:
                    self.logger.warning("Structural LRA is disabled - param_ratio will be low")

                # BLT structural compression was already applied
                blt_structural = state.branch_results.get("blt")
                if blt_structural and blt_structural.metadata.get("structurally_replaced", 0) > 0:
                    self.logger.info("✓ BLT structurally compressed %d embeddings", 
                                   blt_structural.metadata.get("structurally_replaced"))

                evaluation = self._evaluate(state, evaluation_inputs)
                state.evaluation = evaluation.metrics

                state.merged_state = state.bundle.student.state_dict()

                summary = build_compression_summary(self.config, state)
                logging_manager.log_metrics(summary.to_metrics())
                if summary.notes:
                    for note in summary.notes:
                        self.logger.warning(note)

                checkpoint = self.storage.persist(state)

        return PipelineArtifacts(checkpoint=checkpoint, evaluation=evaluation, summary=summary, state=state)

    def _initial_state(self) -> PipelineState:
        bundle = ModelBundle(
            teacher=self.bundle.teacher,
            student=self.bundle.student,
            tokenizer=self.bundle.tokenizer,
            label=self.bundle.label or "distilled",
        )

        return PipelineState(bundle=bundle, work_dir=self.work_dir, device=self.device)

    def _prepare(self, state: PipelineState) -> None:
        self.logger.info("Running baseline analysis (sampling subset for speed)")
        # Sample a subset of layers for faster estimation
        teacher = state.bundle.teacher
        baseline_params = {name: tensor.numel() for name, tensor in iter_named_parameters(state.bundle.student)}
        state.distillation_history["baseline_student_params"] = baseline_params
        state.distillation_history["baseline_student_total"] = sum(baseline_params.values())
        sample_count = 0
        max_samples = 10  # Limit samples for speed
        
        ranks = {}
        energies = {}
        
        for name, tensor in iter_named_parameters(teacher):
            if sample_count >= max_samples:
                break
            if tensor.ndim < 2:
                continue
            if "embed" in name.lower():
                continue
            
            matrix = tensor.reshape(tensor.shape[0], -1).to(torch.float32)
            u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
            cumulative = torch.cumsum(s, dim=0)
            threshold = 0.9 * cumulative[-1]
            rank = int(torch.searchsorted(cumulative, threshold).item() + 1)
            
            ranks[name] = rank
            energies[name] = float(s[:rank].sum() / s.sum())
            sample_count += 1
        
        from .utils.rank_estimation import RankStatistics
        stats = RankStatistics(layer_ranks=ranks, spectral_energy=energies)
        state.distillation_history["baseline_rank"] = ranks
        state.distillation_history["mean_rank"] = stats.mean_rank() if ranks else 0.0
        # Plan per-branch budgets to aim for target ratio
        self._plan_budgets(state)

    def _plan_budgets(self, state: PipelineState) -> None:
        """Heuristically allocate compression budgets to meet target ratio.

        Sets LRA rank, KV cache tokens, BLT latent dim, and sparsity target based on
        model shapes and the configured targets.compression_ratio. This is a static
        planner intended to get close to the target without iterative loops.
        """
        target_ratio = max(float(self.config.targets.compression_ratio), 1.0)
        base_params = count_parameters(state.bundle.teacher)
        if base_params <= 0:
            return

        desired_effective_total = max(int(base_params / target_ratio), 1)

        student = state.bundle.student
        kv_tokens = ("k_proj", "v_proj", "attn", "query", "key", "value")

        embed_shapes: list[tuple[int, int]] = []
        kv_shapes: list[tuple[int, int]] = []
        lra_shapes: list[tuple[int, int]] = []

        embed_original_total = 0
        kv_original_total = 0
        lra_original_total = 0
        total_params = 0

        for name, weight in iter_named_parameters(student):
            total_params += int(weight.numel())
            if weight.ndim < 2:
                continue

            rows = int(weight.shape[0])
            cols = int(torch.tensor(list(weight.shape[1:])).prod().item()) if len(weight.shape) > 2 else int(weight.shape[1])
            key = name.lower()

            if "embed" in key:
                embed_shapes.append((rows, cols))
                embed_original_total += rows * cols
                continue

            if any(tok in key for tok in kv_tokens):
                kv_shapes.append((rows, cols))
                kv_original_total += rows * cols
                continue

            lra_shapes.append((rows, cols))
            lra_original_total += rows * cols

        residual_original_total = max(total_params - embed_original_total - kv_original_total - lra_original_total, 0)

        latent_dim = max(1, self.config.branches.blt.latent_dim)
        cache_tokens = max(1, self.config.branches.kv.cache_tokens)
        lra_rank = max(1, self.config.branches.lra.rank)
        sparsity_target = min(max(self.config.branches.sparsity.target_sparsity, 0.0), 0.999)

        if embed_shapes:
            max_embed_rank = max(min(rows, cols) for rows, cols in embed_shapes)
            latent_dim = min(latent_dim, max_embed_rank)
        if kv_shapes:
            max_kv_rank = max(min(rows, cols) for rows, cols in kv_shapes)
            cache_tokens = min(cache_tokens, max_kv_rank)
        if lra_shapes:
            max_lra_rank = max(min(rows, cols) for rows, cols in lra_shapes)
            lra_rank = min(lra_rank, max_lra_rank)

        def estimate_total(rank: int, cache: int, latent: int, sparsity: float) -> float:
            embed_est = 0
            for rows, cols in embed_shapes:
                if min(rows, cols) <= 1:
                    embed_est += rows * cols
                    continue
                max_rank = max(1, min(rows, cols) - 1)
                target_r = max(1, min(latent, max_rank))
                embed_est += rows * target_r + target_r * cols

            kv_est = 0
            for rows, cols in kv_shapes:
                if min(rows, cols) <= 1:
                    kv_est += rows * cols
                    continue
                max_rank = max(1, min(rows, cols))
                target_r = max(1, min(cache, max_rank))
                kv_est += rows * target_r + target_r * cols

            lra_est = 0
            for rows, cols in lra_shapes:
                if min(rows, cols) <= 1:
                    lra_est += rows * cols
                    continue
                frac_cap = max(1, int(self.config.targets.max_rank_fraction * min(rows, cols)))
                full_cap = max(1, min(rows, cols) - 1)
                max_rank = max(1, min(frac_cap, full_cap))
                target_r = max(1, min(rank, max_rank))
                lra_est += rows * target_r + target_r * cols

            residual_est = residual_original_total * max(1.0 - sparsity, 0.0)
            return float(embed_est + kv_est + lra_est + residual_est)

        expected_total = estimate_total(lra_rank, cache_tokens, latent_dim, sparsity_target)

        for _ in range(5):
            if expected_total <= desired_effective_total:
                break

            scale = max(desired_effective_total / expected_total, 0.05)

            lra_rank = max(1, int(max(lra_rank * scale, 1)))
            cache_tokens = max(1, int(max(cache_tokens * scale, 1)))
            latent_dim = max(1, int(max(latent_dim * scale, 1)))

            density = max(1.0 - sparsity_target, 1e-4)
            density = max(density * scale, 1e-5)
            sparsity_target = min(0.999, 1.0 - density)

            expected_total = estimate_total(lra_rank, cache_tokens, latent_dim, sparsity_target)

        if lra_shapes:
            max_rank_cap = max(
                max(1, int(self.config.targets.max_rank_fraction * min(r, c)))
                for r, c in lra_shapes
            )
            lra_rank = max(1, min(lra_rank, max_rank_cap))

        self.config.branches.lra.rank = int(lra_rank)
        self.config.branches.kv.cache_tokens = int(cache_tokens)
        self.config.branches.blt.latent_dim = int(latent_dim)
        self.config.branches.sparsity.target_sparsity = max(
            float(self.config.branches.sparsity.target_sparsity), float(sparsity_target)
        )
        sparsity_final = float(self.config.branches.sparsity.target_sparsity)

        if embed_original_total > 0 and embed_shapes:
            embed_est = 0
            for rows, cols in embed_shapes:
                if min(rows, cols) <= 1:
                    continue
                max_rank = max(1, min(rows, cols) - 1)
                target_r = max(1, min(self.config.branches.blt.latent_dim, max_rank))
                embed_est += rows * target_r + target_r * cols
            reduction = max(0.0, min(0.999, 1.0 - (embed_est / embed_original_total))) if embed_est else 0.0
            self.config.branches.blt.embedding_reduction = max(
                float(self.config.branches.blt.embedding_reduction), float(reduction)
            )

        needs_structural = target_ratio > 10.0
        if needs_structural and not self.config.branches.lra.enable_structural:
            self.config.branches.lra.enable_structural = True
            self.logger.info(
                "Auto-enabled structural LRA for target ratio %.2fx (threshold: 10.0x)", target_ratio
            )

        expected_total_final = estimate_total(
            self.config.branches.lra.rank,
            self.config.branches.kv.cache_tokens,
            self.config.branches.blt.latent_dim,
            sparsity_final,
        )

        self.logger.info(
            "Planned budgets -> latent_dim=%d, kv_rank=%d, lra_rank=%d, sparsity=%.4f, structural=%s | expected≈%.2feff vs target %d (teacher params %d)",
            int(self.config.branches.blt.latent_dim),
            int(self.config.branches.kv.cache_tokens),
            int(self.config.branches.lra.rank),
            sparsity_final,
            "enabled" if self.config.branches.lra.enable_structural else "disabled",
            expected_total_final,
            desired_effective_total,
            base_params,
        )

    def _run_branches(self, state: PipelineState, logging_manager: LoggingManager) -> None:
        branches = self.registry.create(self.config.mode)
        if not branches:
            self.logger.warning("No branches enabled for mode %s", self.config.mode)
            return

        def execute(branch):
            branch.prepare(state)
            artifact = branch.run(state)
            logging_manager.log_metrics(artifact.metrics, prefix=f"branch/{branch.name}/")
            branch.finalize(state, artifact)
            return branch.name, artifact

        self.logger.info("Executing %d branches", len(branches))
        with ThreadPoolExecutor(max_workers=len(branches)) as executor:
            for name, artifact in executor.map(execute, branches):
                state.branch_results[name] = artifact

    def _merge(self, state: PipelineState) -> None:
        base_state = state.bundle.student.state_dict()
        merged = self.merger.merge(base_state, state.branch_results.values())
        state.bundle.student.load_state_dict(merged, strict=False)
        state.merged_state = merged

    def _distill(
        self,
        state: PipelineState,
        dataloader: torch.utils.data.DataLoader,
        progress: Optional[object] = None,
        task: Optional[object] = None,
    ) -> None:
        optimizer = torch.optim.AdamW(state.bundle.student.parameters(), lr=1e-4)
        report = self.distiller.distill(state, dataloader, optimizer, progress=progress, task=task)
        state.distillation_history["report"] = report.losses

    def _finetune(
        self,
        state: PipelineState,
        dataloader: torch.utils.data.DataLoader,
        progress: Optional[object] = None,
        task: Optional[object] = None,
    ) -> None:
        optimizer = torch.optim.AdamW(state.bundle.student.parameters(), lr=self.config.finetune.learning_rate)
        report = self.finetuner.run(
            state.bundle.student, dataloader, optimizer, device=state.device, progress=progress, task=task
        )
        state.finetune_history = report.history

    def _evaluate(self, state: PipelineState, evaluation_inputs: Optional[EvaluationInputs]) -> EvalResult:
        if evaluation_inputs is None:
            evaluation_inputs = EvaluationInputs()
        return self.evaluator.run(state.bundle.student, evaluation_inputs)

    def _handle_structural_lra(self, state: PipelineState, report: StructuralLRAReport) -> None:
        self.logger.info(
            "✓ Structural LRA replaced %d linear layers (ratio %.2fx)",
            report.replaced,
            report.ratio,
        )
        state.distillation_history["structural_lra_replaced"] = report.replaced
        state.distillation_history["structural_lra_report"] = report

        if not report.original_params:
            return

        replaced_names = set(report.original_params.keys())
        removed = set(state.distillation_history.get("structural_removed_params", []))
        removed.update(replaced_names)
        state.distillation_history["structural_removed_params"] = sorted(removed)

        for artifact in state.branch_results.values():
            metadata = artifact.metadata if artifact.metadata else {}
            effective = metadata.get("effective_params") if metadata else None
            if not effective:
                continue
            for key in list(effective.keys()):
                if key in replaced_names:
                    effective.pop(key, None)

        lra_artifact = state.branch_results.get("lra")
        if lra_artifact:
            lra_artifact.metadata.setdefault("structural_original_params", report.original_params)
            lra_artifact.metadata.setdefault("structural_new_params", report.new_params)
            lra_artifact.metrics["structural_ratio"] = report.ratio

    def _handle_structural_blt(self, state: PipelineState) -> None:
        artifact = state.branch_results.get("blt")
        if not artifact or artifact.metadata is None:
            return

        removed_params = artifact.metadata.get("structural_removed_params") or []
        if not removed_params:
            return

        removed = set(state.distillation_history.get("structural_removed_params", []))
        removed.update(removed_params)
        state.distillation_history["structural_removed_params"] = sorted(removed)

        removed_set = set(removed_params)
        for branch_artifact in state.branch_results.values():
            meta = branch_artifact.metadata
            if not meta:
                continue
            effective = meta.get("effective_params")
            if not effective:
                continue
            for key in list(effective.keys()):
                if key in removed_set:
                    effective.pop(key, None)

        original_map = artifact.metadata.get("structural_original_params") or {}
        new_map = artifact.metadata.get("structural_new_params") or {}
        if original_map and new_map:
            total_new = sum(new_map.values()) or 1
            ratio = sum(original_map.values()) / total_new if sum(original_map.values()) else 1.0
            artifact.metrics["structural_ratio"] = ratio


__all__ = ["CompressionPipeline", "PipelineArtifacts"]


