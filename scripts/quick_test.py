"""Quick test script to verify installation and basic functionality."""

from __future__ import annotations

from pathlib import Path

from distilled_kv import CompressionPipeline, ModelBundle, load_config
from distilled_kv.utils.models import TinyByteLM
import torch
from torch.utils.data import DataLoader, Dataset


class SimpleDataset(Dataset):
    def __len__(self) -> int:
        return 16

    def __getitem__(self, idx: int):
        tokens = torch.randint(0, 32, (32,))
        return {"input_ids": tokens, "labels": tokens.clone()}


def main():
    print("🧪 Quick Test: Distilled KV Compression Pipeline")
    print("=" * 60)

    # Load default config
    config_path = Path("configs") / "default.yaml"
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print("   Please ensure you're running from the project root.")
        return

    print(f"✓ Loading config from {config_path}")
    config = load_config(config_path)

    # Create tiny models for quick test
    # Note: Using same hidden_size to avoid shape mismatches when branches compress dimensions
    # In real usage, you'd design student architecture to match compressed teacher structure
    print("✓ Creating teacher and student models...")
    teacher = TinyByteLM(vocab_size=32, hidden_size=64)
    student = TinyByteLM(vocab_size=32, hidden_size=64)  # Same size for compatibility

    bundle = ModelBundle(teacher=teacher, student=student, label="quick-test")
    print(f"  Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"  Student params: {sum(p.numel() for p in student.parameters()):,}")

    # Initialize pipeline
    print("✓ Initializing compression pipeline...")
    pipeline = CompressionPipeline(config, bundle, device=torch.device("cpu"))

    # Create simple data loaders
    print("✓ Creating data loaders...")
    dataset = SimpleDataset()
    loader = DataLoader(dataset, batch_size=4)

    # Run pipeline
    print("✓ Running compression pipeline...")
    print("  This may take a moment...")
    try:
        artifacts = pipeline.run(
            distill_loader=loader,
            finetune_loader=loader,
        )

        # Display results
        print("\n" + "=" * 60)
        print("✅ SUCCESS!")
        print("=" * 60)
        print(f"Checkpoint: {artifacts.checkpoint}")
        print(f"\nCompression Summary:")
        print(f"  Target ratio: {artifacts.summary.target_ratio:.2f}x")
        print(f"  Effective ratio: {artifacts.summary.effective_ratio:.2f}x")
        print(f"  Meets target: {'✅' if artifacts.summary.meets_target else '❌'}")
        print(f"\nBranch Contributions:")
        for name, ratio in artifacts.summary.branch_ratios.items():
            print(f"  - {name}: {ratio:.2f}x")

        if artifacts.summary.notes:
            print(f"\nNotes:")
            for note in artifacts.summary.notes:
                print(f"  * {note}")

        print("\n" + "=" * 60)
        print("✨ Installation verified! You can now run full experiments.")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ ERROR during pipeline execution:")
        print(f"   {type(e).__name__}: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()

