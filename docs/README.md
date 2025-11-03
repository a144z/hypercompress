# Documentation Index

This folder contains advanced documentation for the Hypercompress library.

## 📚 Documentation Files

### [1000× Compression Playbook](1000xplaybook.md)
**The definitive guide to achieving extreme neural network compression**

A comprehensive playbook covering:
- Understanding 1000× compression metrics and targets
- Step-by-step instructions for your first 1000× run
- Configuration strategies and parameter tuning
- Detailed pipeline execution flow
- Branch-specific tuning (LRA, KV-Distill, BLT, Sparsity)
- Advanced techniques (iterative compression, custom branches)
- Troubleshooting common issues
- Performance benchmarks by model size
- Best practices and real-world examples

**Start here** if you want to achieve extreme compression ratios (100×, 1000×, or beyond).

### [Compression Implementation Plan](compression-plan.md)
**Technical architecture and design decisions**

Documentation covering:
- Root cause analysis of compression challenges
- Pipeline architecture design
- Structural compression phases
- Implementation details

**Read this** if you want to understand how the compression pipeline works internally.

### [Hypercompression Fixes](hypercompression-fixed.md)
**Technical notes on pipeline improvements**

Documentation of:
- Issues that were fixed in the compression pipeline
- BLT branch rewrite details
- Pipeline execution order improvements
- Budget planner enhancements
- Expected results before and after fixes

**Read this** if you want to understand recent improvements and technical fixes.

## 📖 Quick Links

- [Main README](../README.md) - Project overview and quick start
- [Quick Start Guide](../QUICKSTART.md) - Getting started tutorial
- [Configuration Files](../configs/) - Example configuration files

## 🎯 Getting Started

1. **New to Hypercompress?** → Start with [README.md](../README.md) and [QUICKSTART.md](../QUICKSTART.md)
2. **Want to achieve 1000× compression?** → Read [1000× Compression Playbook](1000xplaybook.md)
3. **Understanding the architecture?** → Check [Compression Implementation Plan](compression-plan.md)
4. **Troubleshooting issues?** → See the [Troubleshooting section](1000xplaybook.md#troubleshooting) in the playbook

## 🔧 Contributing

When adding new documentation:
- Place user-facing guides in the root directory (README.md, QUICKSTART.md)
- Place advanced/technical documentation in `docs/`
- Update this index file to include new documentation
- Follow the same markdown style and structure

