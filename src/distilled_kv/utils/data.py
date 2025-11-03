from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset


class TextTokenDataset(Dataset):
    """Creates fixed-length token chunks from raw text files with streaming to handle large files."""

    def __init__(
        self,
        *,
        file_path: str | Path,
        tokenizer,
        seq_len: int,
        stride: int | None = None,
        max_chunks: int | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(self.file_path)

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.max_chunks = max_chunks  # Limit total chunks for faster loading

        # Read file in chunks to avoid loading entire file into memory
        chunk_size_bytes = 1_000_000  # 1MB chunks
        all_chunks: List[torch.Tensor] = []

        with self.file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            buffer = ""
            for chunk_text in iter(lambda: handle.read(chunk_size_bytes), ""):
                buffer += chunk_text

                # Tokenize when we have enough text
                if len(buffer) > chunk_size_bytes * 2:
                    # Tokenize in smaller batches to respect model max length
                    batch_size = min(10000, len(buffer))
                    text_part = buffer[:batch_size]
                    buffer = buffer[batch_size:]

                    try:
                        encoded = tokenizer(
                            text_part,
                            return_tensors="pt",
                            truncation=True,
                            max_length=tokenizer.model_max_length if hasattr(tokenizer, "model_max_length") else 512,
                            add_special_tokens=True,
                        )
                        tokens = encoded["input_ids"].squeeze(0)
                        if tokens.numel() > 0:
                            # Split into sequence-length chunks
                            for start in range(0, max(1, tokens.size(0) - seq_len), self.stride):
                                end = min(start + seq_len, tokens.size(0))
                                if end - start >= seq_len // 2:  # Only keep substantial chunks
                                    all_chunks.append(tokens[start:end])
                                    if self.max_chunks and len(all_chunks) >= self.max_chunks:
                                        break
                    except Exception:
                        continue  # Skip problematic chunks

                if self.max_chunks and len(all_chunks) >= self.max_chunks:
                    break

            # Process remaining buffer
            if buffer and (not self.max_chunks or len(all_chunks) < self.max_chunks):
                try:
                    encoded = tokenizer(
                        buffer,
                        return_tensors="pt",
                        truncation=True,
                        max_length=tokenizer.model_max_length if hasattr(tokenizer, "model_max_length") else 512,
                    )
                    tokens = encoded["input_ids"].squeeze(0)
                    if tokens.numel() > 0:
                        for start in range(0, max(1, tokens.size(0) - seq_len), self.stride):
                            end = min(start + seq_len, tokens.size(0))
                            if end - start >= seq_len // 2:
                                all_chunks.append(tokens[start:end])
                                if self.max_chunks and len(all_chunks) >= self.max_chunks:
                                    break
                except Exception:
                    pass

        if not all_chunks:
            raise ValueError(f"Could not create any valid chunks from {self.file_path}")

        # Pad all chunks to same length
        padded_chunks = []
        for chunk in all_chunks:
            if chunk.size(0) < seq_len:
                padding = torch.zeros(seq_len - chunk.size(0), dtype=chunk.dtype)
                chunk = torch.cat([chunk, padding])
            padded_chunks.append(chunk[:seq_len])

        self.examples = torch.stack(padded_chunks) if len(padded_chunks) > 1 else torch.stack([padded_chunks[0]])

    def __len__(self) -> int:
        return self.examples.size(0)

    def __getitem__(self, idx: int):  # noqa: D401
        tokens = self.examples[idx]
        return {"input_ids": tokens, "labels": tokens.clone()}


__all__ = ["TextTokenDataset"]


