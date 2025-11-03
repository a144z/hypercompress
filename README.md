# Hypercompress

**Extreme-scale neural network compression toolkit** targeting up to **1000× effective model size reduction** through hybrid compression techniques. Hypercompress combines low-rank approximation (LRA), KV-cache distillation, byte latent transformers (BLT), and aggressive sparsification in a unified, production-ready pipeline.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Key Features

- **🧬 Hybrid Compression Pipeline**: Modular branches (LRA, KV-distill, BLT, sparsity) run concurrently with configurable orchestration
- **📊 Adaptive Rank Estimation**: ARSVD with guard-railed thresholds per attention and MoE block
- **🎯 1000× Compression Target**: Automatic tracking and per-branch ratio attribution toward extreme compression goals
- **👨‍🏫 Teacher–Student Distillation**: Hierarchical LoRA updates with activation alignment
- **🔄 Iterative Fine-Tuning**: ReLoRA updates with Adapprox optimizers
- **📈 Comprehensive Evaluation**: Built-in harness for PPL, MMLU, GSM8K, GLUE, and branch ablations
- **📦 Artifact Management**: Integrated Weights & Biases and TensorBoard logging
- **🐳 Production Ready**: Batteries-included Docker images, Taskfile automation, and comprehensive test suite

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Compression Techniques](#compression-techniques)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Completed Tasks](#completed-tasks)
- [To Do List](#to-do-list)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.2+ (with CUDA support recommended)
- For Hugging Face models: `transformers` and `accelerate`

### Install Hypercompress

```bash
# Using pip
pip install -e .

# Or using uv (recommended, faster)
uv pip install -e .

# With development tools (tests, linting)
pip install -e ".[dev]"
```

### Verify Installation

```bash
python scripts/quick_test.py
```

This runs a minimal compression pipeline with synthetic data to ensure everything works correctly.

## 🎯 Quick Start

### Basic Example (Synthetic Data)

Run the compression pipeline with a sample configuration:

```bash
python scripts/run_pipeline.py --config configs/sample_hybrid.yaml --ratio 100
```

### Target 1000× Compression

```bash
python scripts/run_pipeline.py --config configs/default.yaml --ratio 1000
```

### Real-World LLM Compression

Compress large language models from Hugging Face:

```bash
python scripts/run_hf_pipeline.py \
  --config configs/hf_12b.yaml \
  --ratio 100 \
  --teacher-model meta-llama/Meta-Llama-3-12B \
  --student-model meta-llama/Meta-Llama-3-8B-Instruct \
  --distill-data data/wikipedia.txt \
  --finetune-data data/instruct_mix.txt \
  --seq-len 2048 \
  --stride 1024 \
  --batch-size 1
```

**Quick dataset setup:**

```bash
mkdir -p data
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/distill.txt
cp data/distill.txt data/finetune.txt
```

**Public model example (no auth required):**

```bash
python scripts/run_hf_pipeline.py \
  --config configs/hf_example_public.yaml \
  --teacher-model microsoft/phi-2 \
  --student-model microsoft/phi-1_5 \
  --distill-data data/distill.txt \
  --seq-len 2048 \
  --stride 1024
```

For detailed instructions, see [QUICKSTART.md](QUICKSTART.md).

## 🧪 Compression Techniques

Hypercompress implements a multi-stage hybrid compression pipeline:

### 1. Low-Rank Approximation (LRA)
Factorizes weight matrices using singular value decomposition (SVD) to reduce parameter count while preserving performance.

### 2. KV-Cache Distillation
Compresses key-value cache projections to reduce memory footprint during inference.

### 3. Byte Latent Transformers (BLT)
Structural compression of embedding layers by replacing high-dimensional embeddings with low-dimensional latent representations.

### 4. Sparsity Acceleration
Prunes weights to achieve extreme sparsity ratios (up to 99%+) while maintaining model accuracy.

### 5. Structural Compression
Replaces linear layers with factorized equivalents to achieve true parameter reduction.

### 6. Knowledge Distillation
Teacher-student training transfers knowledge from large models to compressed students.

## ⚙️ Configuration

Compression runs are configured via YAML files. Example:

```yaml
mode: hybrid
targets:
  compression_ratio: 1000
  retained_accuracy: 0.95
branches:
  lra:
    rank: 64
    sparsity: 0.99
  kv:
    cache_tokens: 128
  blt:
    embedding_reduction: 0.9
  sparsity:
    target_sparsity: 0.99
logging:
  experiment: sample-hybrid
```

See `configs/default.yaml` for a complete reference of all available options.

## 📊 Compression Reporting

Every pipeline run produces a `CompressionSummary` that aggregates:

- Branch-level compression ratios
- Parameter statistics (before/after)
- Target compliance tracking
- Per-branch attribution metrics

Results are automatically logged to TensorBoard, Weights & Biases, or custom dashboards under the `compression/*` namespace.

## 📁 Project Structure

```
hypercompress/
├─ src/distilled_kv/          # Main package (NOTE: directory name will be updated)
│  ├─ pipeline.py             # Main orchestrator
│  ├─ config.py               # Typed configuration loading
│  ├─ logging.py              # Structured logging utilities
│  ├─ modules/                # Compression branches
│  │  ├─ lra.py
│  │  ├─ kv_distill.py
│  │  ├─ blt.py
│  │  ├─ sparsity.py
│  │  └─ merge.py
│  ├─ structural/             # Structural compression utilities
│  ├─ utils/                   # Shared math + helpers
│  ├─ analysis/                # Compression summaries and reports
│  ├─ distillation/           # Teacher-student distillation
│  ├─ finetune/                # Fine-tuning loops
│  ├─ evaluation/              # Evaluation harness
│  └─ storage/                 # Artifact management
├─ configs/                    # YAML configuration presets
├─ scripts/                     # Runnable entry points
├─ tests/                       # Unit tests (pytest)
├─ docker/                      # Containerized training environment
├─ Taskfile.yml                # Automation helpers
└─ pyproject.toml              # Packaging metadata
```

## 🧪 Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=distilled_kv --cov-report=html
```

## 📚 Advanced Topics

For advanced research directions (ZipNN, chained hierarchical LoRA, recursive SSMs, etc.) that can be incorporated into custom branches, see `docs/1000x-playbook.md`.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests and linting: `Task lint`
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

Please ensure new features include:

- Test coverage
- Documentation updates
- Type hints where applicable

## 📝 License

MIT © 2025 Hypercompress Contributors

## ✅ Completed Tasks

- ✅ Renamed repository from "Distilled KV" to "Hypercompress"
- ✅ Updated README.md with improved structure, badges, and comprehensive documentation
- ✅ Updated `pyproject.toml` with new package name "hypercompress"
- ✅ Updated package docstrings and references to reflect new name
- ✅ Configured comprehensive `.gitignore` to ignore all model files (`.pt`, `.bin`, `.pth`, `.safetensors`, etc.)
- ✅ Configured `.gitignore` to ignore artifacts directories (`.artifacts/`, `artifacts/`, `checkpoints/`)
- ✅ Removed large model files (380+ MB) from git tracking
- ✅ Verified artifacts directory is properly ignored by Git
- ✅ Cleaned up repository cache and verified clean state

## 📝 To Do List

### High Priority
- [ ] Rename package directory from `distilled_kv` to `hypercompress` (requires updating all imports)
- [ ] Update all internal references and documentation to use new package name
- [ ] Add CI/CD pipeline for automated testing
- [ ] Add comprehensive examples in `examples/` directory

### Medium Priority
- [ ] Create detailed API documentation with Sphinx or MkDocs
- [ ] Add more compression technique implementations
- [ ] Benchmark compression performance on standard datasets
- [ ] Improve error handling and user-friendly error messages

### Low Priority
- [ ] Add Docker Compose setup for easy development environment
- [ ] Create video tutorials or interactive notebooks
- [ ] Add support for more model formats (ONNX, TensorFlow)
- [ ] Implement distributed training support

## 🙏 Acknowledgments

Built with cutting-edge research in neural network compression, knowledge distillation, and efficient transformer architectures.
