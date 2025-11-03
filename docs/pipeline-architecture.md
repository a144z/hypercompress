# Compression Pipeline Architecture

**Comprehensive technical documentation of the Hypercompress compression pipeline architecture**

This document provides a detailed overview of how the compression pipeline works, including execution phases, component interactions, data flow, and design principles.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Components](#pipeline-components)
3. [Execution Phases](#execution-phases)
4. [Data Flow](#data-flow)
5. [Compression Branches](#compression-branches)
6. [Budget Planning](#budget-planning)
7. [Design Principles](#design-principles)
8. [State Management](#state-management)
9. [Configuration Flow](#configuration-flow)

---

## Overview

The Hypercompress pipeline implements a **multi-phase, hybrid compression strategy** that combines:

- **Structural compression** (architecture changes)
- **Weight-level compression** (parameter approximation)
- **Knowledge distillation** (teacher-student transfer)
- **Fine-tuning** (accuracy recovery)

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CompressionPipeline                      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Teacher     │  │   Student    │  │  Tokenizer   │    │
│  │   Model      │  │    Model     │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │            BranchRegistry                         │      │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐        │      │
│  │  │ LRA  │  │ KV   │  │ BLT  │  │Sparse│        │      │
│  │  └──────┘  └──────┘  └──────┘  └──────┘        │      │
│  └──────────────────────────────────────────────────┘      │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │  Merge   │  │ Distill  │  │ Finetune │                 │
│  │ Strategy │  │ Distiller│  │  Loop    │                 │
│  └──────────┘  └──────────┘  └──────────┘                 │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ Storage  │  │  Logging │  │Eval Suite│                 │
│  │ Manager  │  │  Manager │  │          │                 │
│  └──────────┘  └──────────┘  └──────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

### Key Concepts

1. **Structural Compression**: Actually changes model architecture (e.g., `Linear(A,B)` → `Linear(A,r) + Linear(r,B)`)
2. **Weight Compression**: Approximates weights without changing structure (e.g., SVD factorization)
3. **Branch System**: Modular compression techniques that run independently
4. **Merge Strategy**: Combines outputs from multiple branches
5. **Pipeline State**: Tracks compression progress and artifacts

---

## Pipeline Components

### Core Components

#### 1. CompressionPipeline
- **Location**: `src/distilled_kv/pipeline.py`
- **Purpose**: Main orchestrator for the compression process
- **Responsibilities**:
  - Initialize teacher/student models
  - Coordinate branch execution
  - Manage pipeline state
  - Orchestrate merge, distillation, and fine-tuning

#### 2. BranchRegistry
- **Location**: `src/distilled_kv/modules/base.py`
- **Purpose**: Manages compression branches
- **Responsibilities**:
  - Register compression branches
  - Create branch instances based on configuration
  - Coordinate concurrent execution

#### 3. Compression Branches
- **LRA Branch**: Low-rank approximation (`src/distilled_kv/modules/lra.py`)
- **KV Branch**: KV-cache distillation (`src/distilled_kv/modules/kv_distill.py`)
- **BLT Branch**: Byte latent transformer (`src/distilled_kv/modules/blt.py`)
- **Sparsity Branch**: Dynamic pruning (`src/distilled_kv/modules/sparsity.py`)

#### 4. MergeStrategy
- **Location**: `src/distilled_kv/modules/merge.py`
- **Purpose**: Combines compressed weights from all branches
- **Algorithm**: Weighted averaging based on compression ratios

#### 5. KnowledgeDistiller
- **Location**: `src/distilled_kv/distillation/distiller.py`
- **Purpose**: Transfers knowledge from teacher to student
- **Loss**: KL divergence + activation matching

#### 6. FineTuner
- **Location**: `src/distilled_kv/finetune/loop.py`
- **Purpose**: Recovers accuracy after compression
- **Optimizer**: AdamW with configurable learning rate

#### 7. EvaluationSuite
- **Location**: `src/distilled_kv/evaluation/suite.py`
- **Purpose**: Evaluates compressed model performance
- **Metrics**: Perplexity, accuracy, task-specific metrics

#### 8. ArtifactManager
- **Location**: `src/distilled_kv/storage/manager.py`
- **Purpose**: Saves and loads model checkpoints
- **Formats**: PyTorch `.pt`, Hugging Face `safetensors`

---

## Execution Phases

The pipeline executes in **8 sequential phases**:

### Phase 1: Baseline Analysis

```python
def _prepare(self, state: PipelineState) -> None:
    # Sample layers for rank estimation
    # Store baseline parameter counts
    # Run budget planner
```

**What happens:**
- Samples up to 10 layers from teacher model
- Computes SVD-based rank estimates
- Stores baseline parameter counts
- Runs budget planner to adjust configuration

**Output:**
- Baseline rank statistics
- Budget-adjusted configuration
- Baseline parameter counts

### Phase 2: Branch Execution

```python
def _run_branches(self, state: PipelineState, logging_manager: LoggingManager):
    # Execute compression branches concurrently
    # BLT runs first (structural changes)
    # LRA, KV, Sparsity run concurrently
```

**Execution Order (Critical!):**

1. **BLT (Byte Latent Transformer)** - Runs FIRST
   ```python
   # Structural compression of embeddings
   Embedding(V, D) → Embedding(V, r) + Linear(r, D)
   ```
   - **Why first**: Must modify structure before other branches
   - **Changes**: Actually replaces embedding modules
   - **Output**: Structurally compressed embeddings

2. **LRA (Low-Rank Approximation)** - Concurrent
   ```python
   # Weight factorization (doesn't change structure yet)
   W ≈ U @ V  # via SVD
   ```
   - **Why concurrent**: Independent weight-level operation
   - **Changes**: Computes factorized weights, doesn't modify model
   - **Output**: Low-rank weight approximations

3. **KV-Distill** - Concurrent
   ```python
   # KV projection compression
   KV_proj → KV_proj_low_rank
   ```
   - **Why concurrent**: Independent operation
   - **Changes**: Compresses KV-related layers
   - **Output**: Compressed KV projections

4. **Sparsity** - Concurrent
   ```python
   # Dynamic pruning
   mask = |weights| > threshold
   weights = weights * mask
   ```
   - **Why concurrent**: Independent pruning operation
   - **Changes**: Creates sparsity masks
   - **Output**: Pruned weight masks

**Branch Output Format:**
```python
BranchArtifact(
    name="branch_name",
    metrics={"ratio": 10.0, "params_reduced": 1000},
    metadata={"effective_params": {...}, "compressed_weights": {...}}
)
```

### Phase 3: Merge Branches

```python
def _merge(self, state: PipelineState) -> None:
    base_state = state.bundle.student.state_dict()
    merged = self.merger.merge(base_state, state.branch_results.values())
    state.bundle.student.load_state_dict(merged, strict=False)
```

**What happens:**
- Takes base student model weights
- Blends compressed weights from all branches
- Uses weighted averaging based on compression ratios
- Loads merged weights into student model

**Merge Algorithm:**
```python
# For each parameter:
merged_weight = (
    branch_weight_1 * ratio_1 +
    branch_weight_2 * ratio_2 +
    ...
) / (ratio_1 + ratio_2 + ...)
```

### Phase 4: Structural Factorization

```python
if self.config.branches.lra.enable_structural:
    report = apply_structural_lra(
        state.bundle.student,
        rank=self.config.branches.lra.rank,
        max_rank_fraction=self.config.targets.max_rank_fraction,
    )
```

**What happens:**
- Replaces `Linear(in, out)` layers with factorized equivalents
- `Linear(in, out)` → `Linear(in, r) + Linear(r, out)`
- **Actually reduces parameter count** (not just approximation)

**Why after merge:**
- Structural changes modify the model graph
- Need weight-level compression merged first
- Then apply structural changes to merged model

**Parameter Reduction:**
```
Before: in * out parameters
After:  (in * r + r * out) parameters
Ratio:  (in * out) / (in * r + r * out)
```

### Phase 5: Knowledge Distillation

```python
def _distill(self, state: PipelineState, dataloader: DataLoader):
    optimizer = torch.optim.AdamW(state.bundle.student.parameters(), lr=1e-4)
    report = self.distiller.distill(state, dataloader, optimizer)
```

**What happens:**
- Teacher model generates logits for input data
- Student model generates logits for same input
- Computes loss: KL divergence + activation matching
- Updates student weights via backpropagation

**Loss Function:**
```python
loss = (
    kl_div(student_logits, teacher_logits) * kl_weight +
    mse(student_activations, teacher_activations) * activation_weight
)
```

### Phase 6: Fine-Tuning

```python
def _finetune(self, state: PipelineState, dataloader: DataLoader):
    optimizer = torch.optim.AdamW(state.bundle.student.parameters(), lr=lr)
    report = self.finetuner.run(state.bundle.student, dataloader, optimizer)
```

**What happens:**
- Task-specific fine-tuning on compressed model
- Standard cross-entropy loss on target task
- Further recovers accuracy lost during compression

**Training Loop:**
```python
for batch in dataloader:
    outputs = model(batch.inputs)
    loss = cross_entropy(outputs.logits, batch.labels)
    loss.backward()
    optimizer.step()
```

### Phase 7: Evaluation

```python
def _evaluate(self, state: PipelineState, evaluation_inputs: EvaluationInputs):
    return self.evaluator.run(state.bundle.student, evaluation_inputs)
```

**What happens:**
- Computes evaluation metrics
- Perplexity, accuracy, task-specific metrics
- Stores results in pipeline state

**Available Metrics:**
- Perplexity (PPL)
- Accuracy
- MMLU (if enabled)
- GSM8K (if enabled)
- GLUE (if enabled)

### Phase 8: Save Checkpoint

```python
checkpoint = self.storage.persist(state)
```

**What happens:**
- Saves complete pipeline state
- Model weights, configuration, metrics
- Optionally exports to Hugging Face format
- Creates compression summary report

**Checkpoint Contents:**
```python
{
    "student_state_dict": {...},
    "config": {...},
    "metrics": {...},
    "compression_summary": {...},
    "distillation_history": {...},
    "finetune_history": {...}
}
```

---

## Data Flow

### Initialization

```
User Input (Config + Models)
    ↓
CompressionPipeline.__init__()
    ↓
Initialize Components:
  - BranchRegistry
  - MergeStrategy
  - KnowledgeDistiller
  - FineTuner
  - EvaluationSuite
  - ArtifactManager
    ↓
Move models to device (CUDA/CPU)
```

### Main Execution Flow

```
PipelineState (initial)
    ↓
Phase 1: Baseline Analysis
    ├─ Sample teacher layers
    ├─ Estimate ranks
    └─ Plan budgets
    ↓
Phase 2: Branch Execution
    ├─ BLT (first, structural)
    ├─ LRA (concurrent)
    ├─ KV (concurrent)
    └─ Sparsity (concurrent)
    ↓
BranchArtifacts (collected)
    ↓
Phase 3: Merge
    └─ Blend branch outputs
    ↓
Merged Student Model
    ↓
Phase 4: Structural LRA
    └─ Replace Linear layers
    ↓
Structurally Compressed Model
    ↓
Phase 5: Knowledge Distillation
    └─ Teacher → Student transfer
    ↓
Phase 6: Fine-Tuning
    └─ Task-specific training
    ↓
Phase 7: Evaluation
    └─ Compute metrics
    ↓
Phase 8: Save Checkpoint
    └─ Persist state
    ↓
PipelineArtifacts (final output)
```

### State Transformation

```
Teacher Model (original)
    ↓
    [Baseline Analysis]
    ↓
Student Model (initial)
    ↓
    [BLT Branch] → Structurally compressed embeddings
    ↓
Student Model (BLT-modified)
    ↓
    [LRA/KV/Sparsity Branches] → Compressed weights
    ↓
BranchArtifacts
    ↓
    [Merge] → Blended weights
    ↓
Student Model (merged)
    ↓
    [Structural LRA] → Factorized layers
    ↓
Student Model (structurally compressed)
    ↓
    [Distillation] → Knowledge transfer
    ↓
    [Fine-tuning] → Accuracy recovery
    ↓
Student Model (final, compressed)
```

---

## Compression Branches

### Branch Interface

All branches implement the `CompressionBranch` interface:

```python
class CompressionBranch(ABC):
    @abstractmethod
    def prepare(self, state: PipelineState) -> None:
        """Setup before compression"""
    
    @abstractmethod
    def run(self, state: PipelineState) -> BranchArtifact:
        """Apply compression, return artifact"""
    
    @abstractmethod
    def finalize(self, state: PipelineState, artifact: BranchArtifact) -> None:
        """Cleanup after compression"""
```

### BLT Branch (Byte Latent Transformer)

**Purpose**: Structural compression of embedding layers

**Implementation**:
```python
# Replace Embedding(V, D) with low-rank version
original_embedding = Embedding(vocab_size=V, embedding_dim=D)
compressed_embedding = Embedding(vocab_size=V, embedding_dim=r)  # r << D
projection = Linear(r, D)

# New forward: compressed_embedding(input) @ projection.weight
```

**Execution Order**: Runs FIRST (must be before other branches)

**Parameter Reduction**:
```
Before: V * D parameters
After:  (V * r + r * D) parameters
Ratio:  (V * D) / (V * r + r * D)
```

### LRA Branch (Low-Rank Approximation)

**Purpose**: Factorize weight matrices using SVD

**Implementation**:
```python
# For each weight matrix W:
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
# Keep top k singular values
U_k = U[:, :k]
S_k = S[:k]
Vh_k = Vh[:k, :]
# Approximate: W ≈ U_k @ diag(S_k) @ Vh_k
```

**Execution Order**: Concurrent with KV and Sparsity

**Parameter Reduction**:
```
Original: rows * cols parameters
Factorized: (rows * rank + rank * cols) parameters
Ratio: (rows * cols) / (rows * rank + rank * cols)
```

### KV-Distill Branch

**Purpose**: Compress key-value projection layers

**Implementation**:
```python
# Find KV-related layers (k_proj, v_proj, query, key, value)
for layer_name, weight in kv_layers:
    # Apply SVD compression
    compressed = svd_compress(weight, target_rank=cache_tokens)
```

**Execution Order**: Concurrent with LRA and Sparsity

**Focus**: Only compresses attention-related projection layers

### Sparsity Branch

**Purpose**: Dynamic pruning to achieve extreme sparsity

**Implementation**:
```python
# Compute magnitude threshold
magnitude = torch.abs(weights)
threshold = torch.quantile(magnitude, target_sparsity)
# Create mask
mask = magnitude > threshold
# Apply mask
pruned_weights = weights * mask
```

**Execution Order**: Concurrent with LRA and KV

**Parameter Reduction**:
```
Before: N parameters
After:  N * (1 - sparsity) parameters (non-zero)
Ratio:  1 / (1 - sparsity)
```

---

## Budget Planning

The budget planner automatically adjusts compression parameters to meet target ratios.

### Algorithm

```python
def _plan_budgets(self, state: PipelineState):
    target_ratio = config.targets.compression_ratio
    base_params = count_parameters(teacher)
    desired_params = base_params / target_ratio
    
    # Analyze model structure
    embed_shapes = find_embedding_layers()
    kv_shapes = find_kv_layers()
    lra_shapes = find_linear_layers()
    
    # Estimate compression with current settings
    expected_params = estimate_total(rank, cache, latent, sparsity)
    
    # Iteratively adjust until target is met
    while expected_params > desired_params:
        scale = desired_params / expected_params
        rank *= scale
        cache *= scale
        latent *= scale
        sparsity = min(0.999, sparsity + (1 - sparsity) * 0.05)
        expected_params = estimate_total(rank, cache, latent, sparsity)
    
    # Apply adjusted settings
    config.branches.lra.rank = rank
    config.branches.kv.cache_tokens = cache
    config.branches.blt.latent_dim = latent
    config.branches.sparsity.target_sparsity = sparsity
```

### Key Features

1. **Automatic Adjustment**: Modifies ranks, cache sizes, latent dims, and sparsity
2. **Model-Aware**: Analyzes actual model structure before planning
3. **Iterative Refinement**: Adjusts until target ratio is achievable
4. **Safety Limits**: Respects `max_rank_fraction` and minimum layer sizes
5. **Auto-Enable Structural**: Enables structural LRA for `ratio > 10.0`

---

## Design Principles

### 1. Separation of Concerns

**Weight Compression vs Structural Changes**:
- Weight compression (branches): Preserves structure, modifies weights
- Structural changes (dedicated steps): Actually reduces parameter count
- Clean separation prevents interference

### 2. Execution Order Matters

**Critical Ordering**:
1. BLT runs FIRST (structural changes to embeddings)
2. Weight compression runs concurrently (independent operations)
3. Merge blends weights
4. Structural LRA runs AFTER merge (final architecture changes)

**Why**: Structural changes must happen before weight compression, but final structural factorization happens after weight merging.

### 3. Modular Branch System

**Benefits**:
- Each branch is independent
- Easy to add new compression techniques
- Branches can run concurrently
- Clear interfaces via `CompressionBranch` ABC

### 4. Progressive Compression

**Phases**:
1. Compress (reduce parameters)
2. Distill (transfer knowledge)
3. Fine-tune (recover accuracy)

**Rationale**: Aggressive compression loses accuracy → Distillation and fine-tuning recover it.

### 5. Configuration-Driven

**Flexibility**:
- All behavior controlled via YAML config
- Budget planner auto-adjusts for targets
- Easy to experiment with different settings

---

## State Management

### PipelineState

The pipeline maintains state throughout execution:

```python
@dataclass
class PipelineState:
    bundle: ModelBundle  # Teacher, student, tokenizer
    work_dir: Path
    device: torch.device
    branch_results: dict[str, BranchArtifact]
    merged_state: dict[str, torch.Tensor]
    distillation_history: dict
    finetune_history: dict
    evaluation: dict
```

### State Flow

```
Initial State
    ↓
[Baseline Analysis] → baseline_rank, baseline_params
    ↓
[Branch Execution] → branch_results
    ↓
[Merge] → merged_state
    ↓
[Structural LRA] → structural_lra_report
    ↓
[Distillation] → distillation_history
    ↓
[Fine-tuning] → finetune_history
    ↓
[Evaluation] → evaluation metrics
    ↓
Final State (saved to checkpoint)
```

### ModelBundle

```python
@dataclass
class ModelBundle:
    teacher: torch.nn.Module
    student: torch.nn.Module
    tokenizer: Optional[PreTrainedTokenizer]
    label: str
```

---

## Configuration Flow

### Configuration Loading

```
YAML Config File
    ↓
load_config() → PipelineConfig (Pydantic model)
    ↓
Validation & Defaults Applied
    ↓
PipelineConfig passed to CompressionPipeline
    ↓
Budget Planner adjusts config.branches.*
    ↓
Final configuration used throughout pipeline
```

### Configuration Structure

```yaml
mode: hybrid  # Which branches to enable
targets:
  compression_ratio: 1000  # Target compression ratio
  max_rank_fraction: 0.01  # Maximum rank as fraction
branches:
  lra:
    rank: 8
    enable_structural: true
  kv:
    cache_tokens: 16
  blt:
    embedding_reduction: 0.95
    latent_dim: 32
  sparsity:
    target_sparsity: 0.99
```

### Budget Planner Modifications

The budget planner can modify:
- `branches.lra.rank`
- `branches.kv.cache_tokens`
- `branches.blt.latent_dim`
- `branches.sparsity.target_sparsity`
- `branches.lra.enable_structural` (auto-enabled for high ratios)

---

## Summary

The Hypercompress pipeline implements a sophisticated multi-phase compression system:

1. **Baseline Analysis**: Understands model structure, plans compression
2. **Branch Execution**: Applies multiple compression techniques concurrently
3. **Merge**: Blends compressed weights intelligently
4. **Structural Factorization**: Actually reduces parameter count
5. **Distillation & Fine-tuning**: Recovers accuracy

Key design principles:
- **Separation of concerns**: Weight vs structural compression
- **Execution order matters**: BLT first, structural LRA last
- **Modular branches**: Easy to extend with new techniques
- **Configuration-driven**: Flexible and experiment-friendly

For usage examples, see:
- [1000× Compression Playbook](1000xplaybook.md) - Step-by-step guide
- [Compression Implementation Plan](compression-plan.md) - Technical details
- [README](../README.md) - Quick start

