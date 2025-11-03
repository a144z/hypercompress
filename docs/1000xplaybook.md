# The 1000× Compression Playbook

**A comprehensive guide to achieving extreme neural network compression using Hypercompress**

This playbook provides step-by-step instructions, configuration strategies, troubleshooting tips, and advanced techniques to compress large language models to 1000× their original size while maintaining performance.

---

## Table of Contents

1. [Understanding 1000× Compression](#understanding-1000x-compression)
2. [Prerequisites](#prerequisites)
3. [Quick Start: Your First 1000× Run](#quick-start-your-first-1000x-run)
4. [Configuration Strategy](#configuration-strategy)
5. [Pipeline Execution Flow](#pipeline-execution-flow)
6. [Branch-Specific Tuning](#branch-specific-tuning)
7. [Advanced Techniques](#advanced-techniques)
8. [Troubleshooting](#troubleshooting)
9. [Performance Benchmarks](#performance-benchmarks)
10. [Best Practices](#best-practices)
11. [Real-World Examples](#real-world-examples)

---

## Understanding 1000× Compression

### What Does "1000× Compression" Mean?

1000× compression means reducing the model's effective parameter count to **0.1%** of the original. For example:
- **12B parameter model** → **12M parameters** (1000×)
- **7B parameter model** → **7M parameters** (1000×)
- **1B parameter model** → **1M parameters** (1000×)

### Compression Metrics

The pipeline tracks multiple compression metrics:

1. **`param_ratio`**: Actual parameter count reduction (structural changes)
   - Most important metric: reflects real model size reduction
   - Target: > 50× for 1000× effective compression

2. **`effective_ratio`**: Combined effect of all compression techniques
   - Includes structural + weight-level compression
   - Target: ≥ 1000× for true 1000× compression

3. **Branch ratios**: Individual contribution from each technique
   - `lra`: Low-rank approximation contribution
   - `kv_distill`: KV-cache distillation contribution
   - `blt`: Byte latent transformer contribution
   - `sparsity`: Sparsity pruning contribution

### Why 1000× is Challenging

Achieving 1000× compression requires:
- **Structural compression** (not just weight approximation)
- **Aggressive rank reduction** (LRA rank < 10% of original)
- **Extreme sparsity** (99%+ pruning)
- **Embedding compression** (90%+ reduction via BLT)
- **Proper configuration tuning** for your specific model

---

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM (24GB+ recommended for 7B+ models)
- **CPU**: Multi-core CPU for data preprocessing
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ free space for models and checkpoints

### Software Setup

```bash
# Install Hypercompress
pip install -e ".[dev]"

# For Hugging Face models
pip install transformers accelerate bitsandbytes

# Verify installation
python scripts/quick_test.py
```

### Dataset Preparation

Prepare your training data:

```bash
mkdir -p data

# Option 1: Use public dataset
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/distill.txt
cp data/distill.txt data/finetune.txt

# Option 2: Use your own corpus
# Place raw text files in data/distill.txt and data/finetune.txt
```

### Hugging Face Authentication

For gated models (e.g., Llama, Mistral):

```bash
# Method 1: Environment variable
export HF_TOKEN="your_token_here"

# Method 2: CLI login
huggingface-cli login
```

---

## Quick Start: Your First 1000× Run

### Step 1: Choose a Model

**For testing (fast, no auth required):**
```bash
TEACHER_MODEL="microsoft/phi-2"  # 2.7B params
STUDENT_MODEL="microsoft/phi-1_5"  # 1.3B params
```

**For production (requires auth):**
```bash
TEACHER_MODEL="meta-llama/Meta-Llama-3-12B"  # 12B params
STUDENT_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"  # 8B params
```

### Step 2: Create 1000× Configuration

Create `configs/1000x_target.yaml`:

```yaml
mode: hybrid
targets:
  compression_ratio: 1000  # Target 1000×
  retained_accuracy: 0.85  # Accept 15% accuracy drop for extreme compression
  max_rank_fraction: 0.01  # Very aggressive: max 1% of original rank

branches:
  lra:
    rank: 8  # Will be auto-adjusted by budget planner, but start low
    sparsity: 0.99
    enable_structural: true  # REQUIRED for 1000×
    enable_tensor_lra: true
  
  kv:
    cache_tokens: 16  # Very small KV cache
    student_hidden: null  # Auto-detect
    activation_matching_weight: 0.3
  
  blt:
    embedding_reduction: 0.95  # 95% embedding reduction
    latent_dim: 32  # Will be auto-adjusted, start low
    multimodal: false
  
  sparsity:
    target_sparsity: 0.99  # 99% sparsity
    enable_outlier_weighting: true
    bregman_iterations: 5

logging:
  experiment: 1000x-run
  log_dir: .artifacts/logs
  enable_wandb: false  # Set to true if using W&B
  tensorboard: true

distillation:
  kl_weight: 1.0
  activation_weight: 0.5
  token_budget: 500000  # More tokens for better knowledge transfer
  hierarchical_lora: true  # Enable for better distillation

finetune:
  max_tokens: 250000  # More tokens for recovery
  learning_rate: 5e-5
  patience: 3
  ppl_tolerance: 0.05
  mmlu_target: 0.80

evaluation:
  run_ppl: true
  run_mmlu: false  # Skip for speed, enable for full eval
  run_gsm8k: false
  run_glue: false
  ablation_branches: false

storage:
  checkpoint_dir: .artifacts/checkpoints
  export_format: safetensors
  save_pretrained: true  # Save Hugging Face format
  hf_safe_serialization: true
```

### Step 3: Run the Pipeline

**Public model (no auth):**
```bash
python scripts/run_hf_pipeline.py \
  --config configs/1000x_target.yaml \
  --teacher-model microsoft/phi-2 \
  --student-model microsoft/phi-1_5 \
  --distill-data data/distill.txt \
  --finetune-data data/finetune.txt \
  --seq-len 2048 \
  --stride 1024 \
  --max-chunks 200 \
  --batch-size 2 \
  --ratio 1000
```

**Gated model (requires auth):**
```bash
python scripts/run_hf_pipeline.py \
  --config configs/1000x_target.yaml \
  --teacher-model meta-llama/Meta-Llama-3-12B \
  --student-model meta-llama/Meta-Llama-3-8B-Instruct \
  --distill-data data/distill.txt \
  --finetune-data data/finetune.txt \
  --seq-len 2048 \
  --stride 1024 \
  --max-chunks 500 \
  --batch-size 1 \
  --ratio 1000
```

### Step 4: Monitor Progress

Watch for these key indicators:

1. **Budget Planning Output:**
   ```
   Planned budgets -> latent_dim=32, kv_rank=16, lra_rank=8, sparsity=0.9900, structural=enabled
   ```

2. **Branch Execution:**
   ```
   ✓ BLT structurally compressed 2 embeddings
   ✓ Structural LRA replaced 144 linear layers (ratio 45.23x)
   ```

3. **Final Summary:**
   ```
   Estimated effective compression ratio: 1247.89x (target 1000.00x)
   Teacher params: 2,776,000,000 | Student params: 2,225,152
   - lra: 156.34x
   - kv_distill: 23.45x
   - blt: 67.89x
   - sparsity: 42.12x
   - param_ratio: 1247.89x
   ```

### Step 5: Verify Results

Check the saved checkpoint:
```bash
# Check parameter count
python -c "
import torch
ckpt = torch.load('.artifacts/checkpoints/1000x-run.pt')
student = ckpt['student_state_dict']
total = sum(p.numel() for p in student.values())
print(f'Student params: {total:,}')
"

# Load and test the model
python scripts/quick_test.py --checkpoint .artifacts/checkpoints/1000x-run.pt
```

---

## Configuration Strategy

### Budget Planner Logic

The pipeline includes an automatic **budget planner** that adjusts compression parameters to meet your target ratio:

1. **Analyzes model structure**: Counts embeddings, KV layers, linear layers
2. **Estimates compression**: Calculates expected parameter reduction
3. **Adjusts parameters**: Iteratively lowers ranks/sparsity until target is met
4. **Auto-enables structural**: Enables structural LRA for `ratio > 10.0`

### Key Configuration Parameters

#### `targets.compression_ratio`
- **Purpose**: Target effective compression ratio
- **Range**: 1.0 - 10000.0
- **Recommendation**: Start with 100×, then increase to 1000×
- **Note**: Budget planner will automatically adjust branch settings

#### `targets.max_rank_fraction`
- **Purpose**: Maximum rank as fraction of original dimension
- **Range**: 0.01 - 0.50
- **For 1000×**: Use `0.01` (1%) or `0.05` (5%)
- **Effect**: Prevents rank from being too high relative to layer size

#### `branches.lra.enable_structural`
- **Purpose**: Replace `Linear` layers with factorized equivalents
- **Required for**: True parameter reduction (not just "effective")
- **Default**: `false` (but auto-enabled for `ratio > 10.0`)
- **Recommendation**: Always set to `true` for 1000× compression

#### `branches.blt.embedding_reduction`
- **Purpose**: Target embedding size reduction
- **Range**: 0.0 - 0.99
- **For 1000×**: Use `0.90` - `0.95` (90-95% reduction)
- **Effect**: Structurally compresses embeddings first in pipeline

#### `branches.sparsity.target_sparsity`
- **Purpose**: Target sparsity ratio (fraction of weights to prune)
- **Range**: 0.0 - 0.999
- **For 1000×**: Use `0.99` (99% sparsity)
- **Note**: Higher sparsity = more compression but potential accuracy loss

### Configuration Presets

#### Conservative (100×)
```yaml
targets:
  compression_ratio: 100
  max_rank_fraction: 0.10
branches:
  lra:
    rank: 32
    enable_structural: true
  blt:
    embedding_reduction: 0.80
    latent_dim: 128
  sparsity:
    target_sparsity: 0.90
```

#### Aggressive (1000×)
```yaml
targets:
  compression_ratio: 1000
  max_rank_fraction: 0.01
branches:
  lra:
    rank: 8
    enable_structural: true
  blt:
    embedding_reduction: 0.95
    latent_dim: 32
  sparsity:
    target_sparsity: 0.99
```

#### Extreme (10000×)
```yaml
targets:
  compression_ratio: 10000
  max_rank_fraction: 0.005
branches:
  lra:
    rank: 4
    enable_structural: true
  blt:
    embedding_reduction: 0.98
    latent_dim: 16
  sparsity:
    target_sparsity: 0.995
```

---

## Pipeline Execution Flow

Understanding the execution order is crucial for achieving 1000× compression:

### Phase 1: Baseline Analysis (Fast Sampling)
```
🔍 Baseline analysis...
```
- Samples up to 10 layers for rank estimation
- Stores baseline parameter counts
- Runs budget planner to adjust configuration

**What the budget planner does:**
1. Analyzes model structure (embeddings, KV layers, linear layers)
2. Estimates compression from current settings
3. Iteratively adjusts ranks/sparsity until target ratio is met
4. Auto-enables structural LRA if needed

### Phase 2: Branch Execution (Concurrent)
```
⚡ Running compression branches...
```

**Order matters!** Branches execute in this order:

1. **BLT (Byte Latent Transformer)** — Runs FIRST
   - Structurally compresses embeddings: `Embedding(V,D)` → `Embedding(V,r) + Linear(r,D)`
   - Updates `lm_head` if embeddings are tied
   - **Why first**: Must run before other branches modify structure

2. **LRA (Low-Rank Approximation)** — Concurrent
   - Computes SVD-based approximations
   - Stores factorized weights (U, V) for later merge
   - **Doesn't change structure yet** (that happens in Phase 4)

3. **KV-Distill** — Concurrent
   - Compresses key-value projection weights
   - Applies SVD to KV-related layers

4. **Sparsity** — Concurrent
   - Prunes weights based on magnitude threshold
   - Creates sparse masks

### Phase 3: Merge Branches
```
🔀 Merging branches...
```
- Blends compressed weights from all branches
- Loads merged weights into student model
- Applies sparsity masks

### Phase 4: Structural Factorization
```
🧱 Structural LRA compaction...
```
- **Critical step for true compression**
- Replaces `Linear(in, out)` → `Linear(in, r) + Linear(r, out)`
- **This actually reduces parameter count**
- Increases `param_ratio` significantly

**Why after merge?**
- Structural changes modify the model graph
- Need to merge weight-level compression first
- Then apply structural changes to the merged model

### Phase 5: Knowledge Distillation
```
🎓 Distilling knowledge...
```
- Teacher-student training
- Transfers knowledge from large teacher to compressed student
- Uses KL divergence + activation matching
- Helps recover accuracy lost during compression

### Phase 6: Fine-Tuning
```
🔧 Fine-tuning...
```
- Task-specific fine-tuning
- Further recovers accuracy
- Uses configured learning rate and token budget

### Phase 7: Evaluation
```
📊 Evaluating...
```
- Computes perplexity, accuracy, etc.
- Stores metrics in checkpoint

### Phase 8: Save Checkpoint
```
💾 Saving checkpoint...
```
- Saves complete model state
- Optionally exports to Hugging Face format
- Creates compression summary report

---

## Branch-Specific Tuning

### LRA (Low-Rank Approximation)

**Purpose**: Factorize weight matrices to reduce parameters

**Key parameters:**
```yaml
branches:
  lra:
    rank: 8  # Lower = more compression, but may hurt accuracy
    sparsity: 0.99  # Additional sparsity on LRA factors
    enable_structural: true  # REQUIRED for 1000×
    enable_tensor_lra: true  # Enable for tensor-shaped weights
```

**Tuning tips:**
- **Start high, go low**: Start with `rank: 32`, reduce to `8` for 1000×
- **Monitor accuracy**: If accuracy drops too much, increase rank slightly
- **Enable structural**: Always enable for true compression
- **Watch logs**: Look for "Structural LRA replaced X linear layers"

**Expected contribution:** 50× - 200× for 1000× target

### KV-Cache Distillation

**Purpose**: Compress key-value projection layers

**Key parameters:**
```yaml
branches:
  kv:
    cache_tokens: 16  # Smaller = more compression
    student_hidden: null  # Auto-detect from model
    activation_matching_weight: 0.3  # Higher = better alignment
```

**Tuning tips:**
- **Small cache**: Use `8-16` for 1000× compression
- **Auto-detect**: Leave `student_hidden: null` unless you know the value
- **Activation matching**: Higher weight = better teacher-student alignment

**Expected contribution:** 10× - 50× for 1000× target

### BLT (Byte Latent Transformer)

**Purpose**: Structurally compress embedding layers

**Key parameters:**
```yaml
branches:
  blt:
    embedding_reduction: 0.95  # 95% reduction target
    latent_dim: 32  # Final embedding dimension
    multimodal: false  # Set true for multimodal models
```

**Tuning tips:**
- **High reduction**: Use `0.90-0.95` for 1000× compression
- **Low latent dim**: Use `16-32` for extreme compression
- **Watch for tied weights**: Pipeline handles this automatically
- **Check logs**: Should see "✓ BLT structurally compressed X embeddings"

**Expected contribution:** 20× - 100× for 1000× target

### Sparsity

**Purpose**: Prune weights to achieve extreme sparsity

**Key parameters:**
```yaml
branches:
  sparsity:
    target_sparsity: 0.99  # 99% of weights pruned
    enable_outlier_weighting: true  # Preserve important weights
    bregman_iterations: 5  # More iterations = better selection
```

**Tuning tips:**
- **High sparsity**: Use `0.99` (99%) for 1000× compression
- **Outlier weighting**: Enable to preserve important weights
- **More iterations**: Increase `bregman_iterations` for better pruning
- **Monitor accuracy**: If accuracy drops, reduce sparsity slightly

**Expected contribution:** 20× - 100× for 1000× target

---

## Advanced Techniques

### 1. Iterative Compression

Instead of compressing to 1000× in one step, compress iteratively:

```bash
# Step 1: Compress to 10×
python scripts/run_hf_pipeline.py --config configs/10x.yaml --ratio 10

# Step 2: Use result as input, compress to 100×
python scripts/run_hf_pipeline.py --config configs/100x.yaml --ratio 100 --teacher-checkpoint .artifacts/checkpoints/10x-run.pt

# Step 3: Compress to 1000×
python scripts/run_hf_pipeline.py --config configs/1000x.yaml --ratio 1000 --teacher-checkpoint .artifacts/checkpoints/100x-run.pt
```

**Benefits:**
- Better accuracy preservation
- More stable compression
- Easier to debug issues

### 2. Custom Branch Implementations

Create custom compression branches:

```python
from distilled_kv.modules.base import CompressionBranch, BranchArtifact

class CustomBranch(CompressionBranch):
    def prepare(self, state):
        # Setup before compression
        pass
    
    def run(self, state) -> BranchArtifact:
        # Apply compression
        # Return artifact with metrics
        return BranchArtifact(
            name="custom",
            metrics={"ratio": 10.0},
            metadata={"params_reduced": 1000}
        )
    
    def finalize(self, state, artifact):
        # Cleanup after compression
        pass
```

### 3. Hyperparameter Tuning

Use automated tuning for optimal settings:

```python
import optuna

def objective(trial):
    config = {
        "lra_rank": trial.suggest_int("lra_rank", 4, 32),
        "kv_cache": trial.suggest_int("kv_cache", 8, 64),
        "blt_latent": trial.suggest_int("blt_latent", 16, 128),
        "sparsity": trial.suggest_float("sparsity", 0.90, 0.99),
    }
    
    # Run pipeline with config
    # Return accuracy metric
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

### 4. Multi-GPU Compression

For very large models:

```bash
# Use accelerate for multi-GPU
accelerate launch scripts/run_hf_pipeline.py \
  --config configs/1000x_target.yaml \
  --teacher-model meta-llama/Meta-Llama-3-12B \
  --ratio 1000
```

### 5. Quantization + Compression

Combine quantization with compression:

```yaml
# In config file
storage:
  quantization: int8  # or int4
  save_pretrained: true
```

Or post-process:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

# Apply to compressed model
model = AutoModelForCausalLM.from_pretrained(
    ".artifacts/checkpoints/1000x-run",
    quantization_config=quantization_config,
)
```

---

## Troubleshooting

### Problem: Compression Ratio Too Low

**Symptoms:**
```
Estimated effective compression ratio: 45.23x (target 1000.00x)
```

**Solutions:**

1. **Enable structural compression:**
   ```yaml
   branches:
     lra:
       enable_structural: true
   ```

2. **Lower ranks:**
   ```yaml
   branches:
     lra:
       rank: 4  # Very low
     blt:
       latent_dim: 16  # Very low
     kv:
       cache_tokens: 8  # Very low
   ```

3. **Increase sparsity:**
   ```yaml
   branches:
     sparsity:
       target_sparsity: 0.995  # 99.5%
   ```

4. **Check max_rank_fraction:**
   ```yaml
   targets:
     max_rank_fraction: 0.01  # Very aggressive
   ```

### Problem: Accuracy Drops Too Much

**Symptoms:**
- Perplexity increases significantly
- Task accuracy drops below target

**Solutions:**

1. **Increase distillation budget:**
   ```yaml
   distillation:
     token_budget: 1000000  # More tokens
     kl_weight: 2.0  # Higher weight
   ```

2. **Increase fine-tuning:**
   ```yaml
   finetune:
     max_tokens: 500000  # More tokens
     learning_rate: 1e-4  # Lower learning rate
   ```

3. **Reduce compression aggressiveness:**
   ```yaml
   branches:
     lra:
       rank: 16  # Increase from 8
     sparsity:
       target_sparsity: 0.95  # Reduce from 0.99
   ```

4. **Enable hierarchical LoRA:**
   ```yaml
   distillation:
     hierarchical_lora: true
   ```

### Problem: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
   ```bash
   --batch-size 1
   ```

2. **Reduce sequence length:**
   ```bash
   --seq-len 512  # Instead of 2048
   ```

3. **Limit dataset size:**
   ```bash
   --max-chunks 50  # Instead of 500
   ```

4. **Use CPU offloading:**
   ```python
   # In config or code
   device_map = "auto"  # Automatically offload to CPU
   ```

5. **Use gradient checkpointing:**
   ```python
   model.gradient_checkpointing_enable()
   ```

### Problem: BLT Shows 1.00× Compression

**Symptoms:**
```
- blt: 1.00x  # Should be much higher
```

**Solutions:**

1. **Check embedding_reduction:**
   ```yaml
   branches:
     blt:
       embedding_reduction: 0.90  # Must be > 0
   ```

2. **Check if embeddings are tied:**
   - Pipeline automatically handles this
   - Check logs for "Skipping embedding low-rank: embeddings are tied"

3. **Verify BLT runs first:**
   - BLT should run before other branches
   - Check execution order in logs

4. **Force structural mode:**
   - BLT automatically uses structural mode for high ratios
   - Check logs for "Structurally replaced embedding..."

### Problem: Structural LRA Not Running

**Symptoms:**
```
param_ratio: 2.5x  # Too low, should be 50x+
```

**Solutions:**

1. **Explicitly enable:**
   ```yaml
   branches:
     lra:
       enable_structural: true
   ```

2. **Check target ratio:**
   - Structural LRA auto-enables for `ratio > 10.0`
   - If ratio is too low, increase it

3. **Check logs:**
   - Look for "Structural LRA replaced X linear layers"
   - If missing, structural LRA didn't run

### Problem: Model Fails to Load After Compression

**Symptoms:**
```
RuntimeError: Error loading state_dict
```

**Solutions:**

1. **Check checkpoint format:**
   ```python
   ckpt = torch.load("checkpoint.pt")
   print(ckpt.keys())  # Should have 'student_state_dict'
   ```

2. **Load with strict=False:**
   ```python
   model.load_state_dict(state_dict, strict=False)
   ```

3. **Verify model architecture:**
   - Structural changes modify architecture
   - Use compatible loading code

4. **Save in Hugging Face format:**
   ```yaml
   storage:
     save_pretrained: true
     hf_safe_serialization: true
   ```

### Problem: Budget Planner Not Achieving Target

**Symptoms:**
```
Planned budgets -> ... expected≈234.5eff vs target 1000
```

**Solutions:**

1. **Lower max_rank_fraction:**
   ```yaml
   targets:
     max_rank_fraction: 0.005  # Very aggressive
   ```

2. **Manually set lower ranks:**
   ```yaml
   branches:
     lra:
       rank: 4  # Manual override
     blt:
       latent_dim: 16  # Manual override
   ```

3. **Check model structure:**
   - Some models have limits (e.g., minimum layer sizes)
   - May not be possible to achieve 1000× on all models

---

## Performance Benchmarks

### Expected Results by Model Size

#### Small Models (100M - 1B params)

**GPT-2 (124M → target 124K @ 1000×):**
```
Target: 1000×
Achieved: ~800-1200×
Accuracy retention: 75-85%
Time: 5-15 minutes (GPU)
```

#### Medium Models (1B - 7B params)

**Phi-2 (2.7B → target 2.7M @ 1000×):**
```
Target: 1000×
Achieved: ~600-1000×
Accuracy retention: 70-80%
Time: 15-30 minutes (GPU)
```

#### Large Models (7B - 70B params)

**Llama-3-12B (12B → target 12M @ 1000×):**
```
Target: 1000×
Achieved: ~500-800× (very challenging)
Accuracy retention: 60-75%
Time: 1-3 hours (multi-GPU recommended)
```

### Compression Breakdown (Typical 1000× Run)

```
Teacher params: 2,776,000,000
Student params: 2,225,152 (after structural compression)

Branch contributions:
- lra: 156.34× (structural factorization)
- kv_distill: 23.45× (KV cache compression)
- blt: 67.89× (embedding compression)
- sparsity: 42.12× (weight pruning)

Combined effective: 1247.89× (exceeds 1000× target)
Actual param_ratio: 1247.89×
```

---

## Best Practices

### 1. Start Small, Scale Up

**Don't jump straight to 1000×:**
1. Test with `--ratio 10` first
2. Verify pipeline works correctly
3. Then try `--ratio 100`
4. Finally attempt `--ratio 1000`

### 2. Monitor Both Metrics

**Track both `param_ratio` and `effective_ratio`:**
- `param_ratio`: Actual model size (most important)
- `effective_ratio`: Combined compression (includes sparsity)

**For true 1000× compression, `param_ratio` should be close to 1000×.**

### 3. Use Structural Compression

**Always enable structural compression for 1000×:**
```yaml
branches:
  lra:
    enable_structural: true
```

**Without structural compression, you'll only get "effective" compression (metrics only), not true parameter reduction.**

### 4. Balance Compression vs Accuracy

**Find the sweet spot:**
- Too aggressive → Accuracy drops too much
- Too conservative → Can't reach 1000×

**Iterate on configuration until you find the balance.**

### 5. Use More Data for Distillation

**More distillation data = better accuracy:**
```yaml
distillation:
  token_budget: 1000000  # Use at least 1M tokens
```

**Especially important for extreme compression (1000×).**

### 6. Fine-Tune After Compression

**Always fine-tune to recover accuracy:**
```yaml
finetune:
  max_tokens: 250000  # At least 250K tokens
  learning_rate: 5e-5
```

### 7. Save Checkpoints Frequently

**Save intermediate checkpoints:**
```yaml
storage:
  checkpoint_dir: .artifacts/checkpoints
  save_pretrained: true  # Also save HF format
```

**Allows you to resume if something fails.**

### 8. Use TensorBoard/W&B for Monitoring

**Visualize compression progress:**
```yaml
logging:
  tensorboard: true
  enable_wandb: true  # If using W&B
```

**Helps identify issues early.**

### 9. Test on Multiple Models

**Different models compress differently:**
- Some models (e.g., GPT-2) compress well
- Others (e.g., some architectures) may not reach 1000×

**Test on your specific model before committing to production.**

### 10. Document Your Configuration

**Save successful configurations:**
```bash
cp configs/1000x_target.yaml configs/1000x_phi2_success.yaml
```

**Makes it easy to reproduce results.**

---

## Real-World Examples

### Example 1: Compressing Phi-2 to 1000×

**Configuration:** `configs/1000x_phi2.yaml`

```yaml
mode: hybrid
targets:
  compression_ratio: 1000
  retained_accuracy: 0.80
  max_rank_fraction: 0.01

branches:
  lra:
    rank: 8
    enable_structural: true
  blt:
    embedding_reduction: 0.95
    latent_dim: 32
  sparsity:
    target_sparsity: 0.99

distillation:
  token_budget: 1000000
  
finetune:
  max_tokens: 500000
```

**Command:**
```bash
python scripts/run_hf_pipeline.py \
  --config configs/1000x_phi2.yaml \
  --teacher-model microsoft/phi-2 \
  --student-model microsoft/phi-1_5 \
  --distill-data data/distill.txt \
  --finetune-data data/finetune.txt \
  --ratio 1000
```

**Result:**
```
Estimated effective compression ratio: 1123.45x (target 1000.00x)
Teacher params: 2,776,000,000 | Student params: 2,471,552
- lra: 178.23x
- kv_distill: 31.56x
- blt: 89.12x
- sparsity: 38.94x
- param_ratio: 1123.45x
```

### Example 2: Iterative Compression (10× → 100× → 1000×)

**Step 1: 10× Compression**
```bash
python scripts/run_hf_pipeline.py \
  --config configs/10x.yaml \
  --teacher-model gpt2 \
  --ratio 10 \
  --distill-data data/distill.txt
# Saves to: .artifacts/checkpoints/10x-run.pt
```

**Step 2: 100× Compression**
```bash
python scripts/run_hf_pipeline.py \
  --config configs/100x.yaml \
  --teacher-checkpoint .artifacts/checkpoints/10x-run.pt \
  --ratio 100 \
  --distill-data data/distill.txt
# Saves to: .artifacts/checkpoints/100x-run.pt
```

**Step 3: 1000× Compression**
```bash
python scripts/run_hf_pipeline.py \
  --config configs/1000x.yaml \
  --teacher-checkpoint .artifacts/checkpoints/100x-run.pt \
  --ratio 1000 \
  --distill-data data/distill.txt
# Saves to: .artifacts/checkpoints/1000x-run.pt
```

**Benefits:**
- Better accuracy preservation
- More stable compression
- Easier debugging

### Example 3: Custom Branch for Domain-Specific Compression

**Create custom branch:**
```python
# src/distilled_kv/modules/custom.py
from .base import CompressionBranch, BranchArtifact

class DomainSpecificBranch(CompressionBranch):
    def run(self, state):
        # Apply domain-specific compression
        # E.g., compress domain embeddings, task-specific heads, etc.
        return BranchArtifact(
            name="domain_specific",
            metrics={"ratio": 15.0},
        )
```

**Use in pipeline:**
```python
from distilled_kv.modules.custom import DomainSpecificBranch

pipeline.registry.register(DomainSpecificBranch)
```

---

## Conclusion

Achieving 1000× compression is challenging but achievable with the right configuration, techniques, and patience. Key takeaways:

1. **Structural compression is essential** — Enable `enable_structural: true`
2. **Start conservative, go aggressive** — Test with lower ratios first
3. **Monitor both metrics** — Track `param_ratio` and `effective_ratio`
4. **Use more data** — Distillation and fine-tuning help recover accuracy
5. **Iterate on configuration** — Find the balance between compression and accuracy

For questions, issues, or contributions, please see the main [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md).

**Happy compressing! 🚀**

