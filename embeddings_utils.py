"""
embeddings_utils.py - Carga de embeddings preentrenados desde archivo local.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def _looks_like_header(parts: list[str]) -> bool:
    if len(parts) != 2:
        return False
    return all(part.isdigit() for part in parts)


def build_embedding_matrix(
    vocab,
    *,
    embeddings_path: str | Path,
    embed_dim: int,
    normalize: bool = False,
) -> tuple[torch.Tensor, dict[str, int | float | str]]:
    path = Path(embeddings_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de embeddings: {path}")

    matrix = torch.empty(len(vocab), embed_dim)
    torch.nn.init.normal_(matrix, mean=0.0, std=0.02)
    matrix[vocab.pad_idx].zero_()

    loaded = 0
    first_data_line = True
    detected_dim = embed_dim

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if first_data_line and _looks_like_header(parts):
                first_data_line = False
                detected_dim = int(parts[1])
                continue

            first_data_line = False
            if len(parts) < 3:
                continue

            token = parts[0]
            values = parts[1:]
            if len(values) != embed_dim:
                detected_dim = len(values)
                continue

            idx = vocab.word2idx.get(token)
            if idx is None and vocab.lowercase:
                idx = vocab.word2idx.get(token.lower())
            if idx is None:
                continue

            vector = torch.tensor([float(value) for value in values], dtype=torch.float32)
            if normalize:
                vector = F.normalize(vector, dim=0)
            matrix[idx] = vector
            loaded += 1

    coverage = loaded / max(len(vocab) - 2, 1)
    stats: dict[str, int | float | str] = {
        "path": str(path),
        "loaded_tokens": loaded,
        "vocab_size": len(vocab),
        "coverage": coverage,
        "detected_dim": detected_dim,
    }
    return matrix, stats


def load_embeddings_into_model(
    model,
    vocab,
    *,
    embeddings_path: str | Path,
    normalize: bool = False,
    freeze: bool = False,
) -> dict[str, int | float | str]:
    matrix, stats = build_embedding_matrix(
        vocab,
        embeddings_path=embeddings_path,
        embed_dim=model.embedding.embedding_dim,
        normalize=normalize,
    )
    with torch.no_grad():
        model.embedding.weight.copy_(matrix)
        model.embedding.weight[vocab.pad_idx].zero_()
    if freeze:
        model.freeze_embedding()
    return stats
