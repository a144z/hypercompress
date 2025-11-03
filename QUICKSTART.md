# Quick Start Guide

## Installation

1. **Install Python 3.10+** (if not already installed)

2. **Install the package in development mode:**
```bash
# Using pip
pip install -e .

# Or using uv (faster)
uv pip install -e .

# With development dependencies (for tests, linting)
pip install -e ".[dev]"
```

## Running the Pipeline

### Basic Example (Synthetic Data)

Run with the sample configuration:

```bash
python scripts/run_pipeline.py --config configs/sample_hybrid.yaml
```

### Custom Configuration

Create your own config file or modify existing ones:

```bash
python scripts/run_pipeline.py --config configs/default.yaml --vocab-size 512 --samples 256
```

### Command Options

```bash
python scripts/run_pipeline.py --help
```

Available options:
- `--config`: Path to YAML configuration file (required)
- `--vocab-size`: Vocabulary size for synthetic data (default: 256)
- `--seq-len`: Sequence length (default: 64)
- `--samples`: Number of synthetic samples (default: 128)
- `--batch-size`: Batch size for training (default: 8)

## Configuration Files

- `configs/default.yaml` - Full default configuration with all options
- `configs/sample_hybrid.yaml` - Minimal hybrid mode example

### Example: Targeting 1000x Compression

Edit `configs/default.yaml` or create a new file:

```yaml
mode: hybrid
targets:
  compression_ratio: 1000  # Target 1000x compression
  retained_accuracy: 0.95
branches:
  lra:
    rank: 64              # Lower rank = more compression
  kv:
    cache_tokens: 128     # Smaller cache = more compression
  blt:
    embedding_reduction: 0.9  # 90% reduction
  sparsity:
    target_sparsity: 0.99  # 99% sparsity
logging:
  experiment: 1000x-test
```

Then run:
```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

### Hugging Face LLMs (12B Example)

```bash
python scripts/run_hf_pipeline.py \
  --config configs/hf_12b.yaml \
  --teacher-model meta-llama/Meta-Llama-3-12B \
  --student-model meta-llama/Meta-Llama-3-8B-Instruct \
  --distill-data data/distill_corpus.txt \
  --finetune-data data/finetune_corpus.txt \
  --seq-len 2048 \
  --stride 1024 \
  --batch-size 1
```

**Requirements**
- Install extras: `pip install transformers accelerate bitsandbytes`
- Ensure Hugging Face auth token is configured if models are gated
- Provide raw text files for distillation / finetune corpora
- Use `--teacher-device-map auto` (default) to let Accelerate shard large models across GPUs/CPU
- Configure `save_pretrained: true` in the storage section to export the final student weights + tokenizer

Example to create a minimal dataset:

```bash
mkdir -p data
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/distill.txt
cp data/distill.txt data/finetune.txt
```

**Public model preset** (no auth token required):

```bash
python scripts/run_hf_pipeline.py \
  --config configs/hf_example_public.yaml \
  --teacher-model microsoft/phi-2 \
  --student-model microsoft/phi-1_5 \
  --distill-data data/distill.txt \
  --seq-len 2048 \
  --stride 1024
```

## Programmatic Usage

```python
from pathlib import Path
from distilled_kv import CompressionPipeline, ModelBundle, load_config
from distilled_kv.utils.models import TinyByteLM
import torch
from torch.utils.data import DataLoader

# Load configuration
config = load_config("configs/sample_hybrid.yaml")

# Create teacher and student models
teacher = TinyByteLM(vocab_size=256, hidden_size=256)
student = TinyByteLM(vocab_size=256, hidden_size=128)

# Create bundle
bundle = ModelBundle(
    teacher=teacher,
    student=student,
    label="my-experiment"
)

# Initialize pipeline
pipeline = CompressionPipeline(config, bundle)

# Create data loaders (or use your own)
# ... your data loading code ...

# Run pipeline
artifacts = pipeline.run(
    distill_loader=your_distill_loader,
    finetune_loader=your_finetune_loader,
)

# Check results
print(f"Compression ratio: {artifacts.summary.effective_ratio:.2f}x")
print(f"Checkpoint saved at: {artifacts.checkpoint}")
```

## Output

After running, you'll find:

- **Checkpoint**: Saved compressed model in `.artifacts/checkpoints/`
- **Logs**: Text logs in `.artifacts/logs/`
- **TensorBoard** (if enabled): `.artifacts/logs/tensorboard/`

The script will print a summary:
```
Compression complete. Checkpoint stored at .artifacts/checkpoints/...
Estimated effective compression ratio: 50.23x (target 1000.00x)
Teacher params: 123,456 | Student params: 45,678
  - lra: 12.34x
  - kv_distill: 8.90x
  - blt: 2.15x
  - sparsity: 5.67x
```

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:
```bash
pytest --cov=distilled_kv --cov-report=html
```

**Note:** The package directory is currently `distilled_kv` (the package will be renamed to `hypercompress` in a future update).

## Troubleshooting

1. **Module not found**: Make sure you ran `pip install -e .`
2. **CUDA out of memory**: Reduce batch size or model sizes in config
3. **Config validation errors**: Check YAML syntax and required fields in `configs/default.yaml`

