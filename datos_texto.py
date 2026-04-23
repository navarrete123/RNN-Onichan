"""
datos_texto.py - Pipeline de datos para clasificacion de texto.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass(frozen=True)
class TextRecord:
    text: str
    label: str


class Vocabulary:
    """Carga vocabularios persistidos y tokeniza texto a ids."""

    def __init__(
        self,
        word2idx: dict[str, int],
        *,
        lowercase: bool = True,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        metadata: dict[str, Any] | None = None,
    ):
        if not word2idx:
            raise ValueError("word2idx no puede estar vacio")

        self.word2idx = {str(token): int(idx) for token, idx in word2idx.items()}
        self.lowercase = lowercase
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.metadata = metadata or {}

        max_idx = max(self.word2idx.values())
        self.idx2word = [""] * (max_idx + 1)
        for token, idx in self.word2idx.items():
            if idx >= len(self.idx2word):
                self.idx2word.extend([""] * (idx - len(self.idx2word) + 1))
            self.idx2word[idx] = token

        self.pad_idx = self.word2idx.get(self.pad_token, 0)
        self.unk_idx = self.word2idx.get(self.unk_token, 1)

    @classmethod
    def load(cls, path: str | Path, *, lowercase: bool = True) -> "Vocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "word2idx" not in payload:
            raise ValueError(f"El archivo {path} no contiene la clave 'word2idx'")
        metadata = {k: v for k, v in payload.items() if k != "word2idx"}
        return cls(payload["word2idx"], lowercase=lowercase, metadata=metadata)

    @classmethod
    def from_state(cls, state: dict[str, Any], *, lowercase: bool = True) -> "Vocabulary":
        if "word2idx" in state:
            metadata = {k: v for k, v in state.items() if k != "word2idx"}
            return cls(state["word2idx"], lowercase=lowercase, metadata=metadata)
        return cls(state, lowercase=lowercase)

    def state_dict(self) -> dict[str, Any]:
        payload = {"word2idx": self.word2idx}
        payload.update(self.metadata)
        return payload

    def __len__(self) -> int:
        return len(self.idx2word)

    def tokenize(self, text: str) -> list[str]:
        if text is None:
            text = ""
        text = str(text)
        if self.lowercase:
            text = text.lower()
        tokens = TOKEN_PATTERN.findall(text)
        return tokens or [self.unk_token]

    def encode_tokens(self, tokens: list[str], max_len: int | None = None) -> list[int]:
        if max_len is not None:
            tokens = tokens[:max_len]
        if not tokens:
            return [self.unk_idx]
        return [self.word2idx.get(token, self.unk_idx) for token in tokens]

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        return self.encode_tokens(self.tokenize(text), max_len=max_len)

    def decode(self, ids: list[int] | torch.Tensor, *, skip_special_tokens: bool = True) -> list[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        tokens: list[str] = []
        special = {self.pad_token, self.unk_token}
        for idx in ids:
            token = self.idx2word[idx] if 0 <= idx < len(self.idx2word) else self.unk_token
            if skip_special_tokens and token in special:
                continue
            tokens.append(token)
        return tokens


class LabelEncoder:
    """Convierte etiquetas de texto a ids consecutivos."""

    def __init__(self, class_names: list[str]):
        if not class_names:
            raise ValueError("Se requiere al menos una clase")
        self.class_names = [str(name) for name in class_names]
        self.label_to_id = {name: idx for idx, name in enumerate(self.class_names)}

    @classmethod
    def fit(cls, labels: list[Any]) -> "LabelEncoder":
        ordered: list[str] = []
        seen: set[str] = set()
        for label in labels:
            label_str = str(label)
            if label_str not in seen:
                ordered.append(label_str)
                seen.add(label_str)
        return cls(ordered)

    @classmethod
    def from_state(cls, state: dict[str, Any] | list[str]) -> "LabelEncoder":
        if isinstance(state, dict):
            if "class_names" in state:
                return cls(state["class_names"])
            if "label_to_id" in state:
                ordered = sorted(state["label_to_id"].items(), key=lambda item: item[1])
                return cls([label for label, _ in ordered])
        return cls(list(state))

    def state_dict(self) -> dict[str, Any]:
        return {
            "class_names": self.class_names,
            "label_to_id": self.label_to_id,
        }

    def encode(self, label: Any) -> int:
        label_str = str(label)
        if label_str not in self.label_to_id:
            raise KeyError(f"Etiqueta desconocida: {label_str}")
        return self.label_to_id[label_str]

    def decode(self, label_id: int) -> str:
        return self.class_names[int(label_id)]

    def __len__(self) -> int:
        return len(self.class_names)


class TextDataset(Dataset):
    """Dataset listo para alimentar al modelo con padding dinamico por batch."""

    def __init__(
        self,
        records: list[TextRecord],
        vocab: Vocabulary,
        label_encoder: LabelEncoder,
        *,
        max_len: int,
    ):
        self.pad_idx = vocab.pad_idx
        self.class_names = label_encoder.class_names
        self.samples: list[tuple[list[int], int, int]] = []

        for record in records:
            token_ids = vocab.encode(record.text, max_len=max_len)
            length = max(1, len(token_ids))
            label_id = label_encoder.encode(record.label)
            self.samples.append((token_ids, length, label_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], int, int]:
        return self.samples[idx]


def collate_text_batch(
    batch: list[tuple[list[int], int, int]],
    *,
    pad_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([length for _, length, _ in batch], dtype=torch.long)
    targets = torch.tensor([label for _, _, label in batch], dtype=torch.long)
    max_batch_len = int(lengths.max().item())
    tokens = torch.full((len(batch), max_batch_len), pad_idx, dtype=torch.long)

    for row, (token_ids, _, _) in enumerate(batch):
        seq = torch.tensor(token_ids[:max_batch_len], dtype=torch.long)
        tokens[row, : seq.numel()] = seq

    return tokens, lengths, targets


def build_text_loader(dataset: TextDataset, cfg, *, shuffle: bool) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.num_workers,
        "collate_fn": partial(collate_text_batch, pad_idx=dataset.pad_idx),
    }
    if cfg.device_type == "cuda":
        kwargs["pin_memory"] = cfg.pin_memory
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def _resolve_column(fieldnames: list[str], requested: str) -> str:
    normalized = {
        str(name).strip().lstrip("\ufeff").lower(): str(name)
        for name in fieldnames
    }
    key = requested.strip().lower()
    if key not in normalized:
        raise ValueError(
            f"No se encontro la columna '{requested}'. Columnas disponibles: {fieldnames}"
        )
    return normalized[key]


def _load_from_delimited_file(
    path: Path,
    *,
    text_column: str,
    label_column: str,
) -> list[TextRecord]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"El archivo {path} no contiene encabezados")
        text_key = _resolve_column(reader.fieldnames, text_column)
        label_key = _resolve_column(reader.fieldnames, label_column)
        records = [
            TextRecord(text=row.get(text_key, ""), label=str(row.get(label_key, "")))
            for row in reader
            if row.get(text_key) is not None and row.get(label_key) is not None
        ]
    if not records:
        raise ValueError(f"No se encontraron filas validas en {path}")
    return records


def _load_from_json_file(
    path: Path,
    *,
    text_column: str,
    label_column: str,
) -> list[TextRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            payload = payload["data"]
        else:
            raise ValueError(
                f"El JSON {path} debe ser una lista de objetos o tener una clave 'data'"
            )
    if not isinstance(payload, list):
        raise ValueError(f"El archivo {path} no contiene una lista de ejemplos")

    records: list[TextRecord] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if text_column not in item or label_column not in item:
            continue
        records.append(TextRecord(text=str(item[text_column]), label=str(item[label_column])))

    if not records:
        raise ValueError(f"No se encontraron ejemplos validos en {path}")
    return records


def _load_from_jsonl_file(
    path: Path,
    *,
    text_column: str,
    label_column: str,
) -> list[TextRecord]:
    records: list[TextRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if text_column not in item or label_column not in item:
                raise ValueError(
                    f"Faltan columnas requeridas en {path} linea {line_number}: "
                    f"'{text_column}' y '{label_column}'"
                )
            records.append(TextRecord(text=str(item[text_column]), label=str(item[label_column])))

    if not records:
        raise ValueError(f"No se encontraron ejemplos validos en {path}")
    return records


def _load_from_class_directories(path: Path) -> list[TextRecord]:
    class_dirs = [item for item in sorted(path.iterdir()) if item.is_dir()]
    if not class_dirs:
        raise ValueError(
            f"La carpeta {path} debe contener subcarpetas por clase con archivos .txt"
        )

    records: list[TextRecord] = []
    for class_dir in class_dirs:
        text_files = sorted(class_dir.rglob("*.txt"))
        for file_path in text_files:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            records.append(TextRecord(text=text, label=class_dir.name))

    if not records:
        raise ValueError(f"No se encontraron archivos .txt dentro de {path}")
    return records


def load_text_records(
    path: str | Path,
    *,
    text_column: str = "text",
    label_column: str = "label",
) -> list[TextRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe la ruta de datos: {path}")

    if path.is_dir():
        return _load_from_class_directories(path)

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return _load_from_delimited_file(
            path, text_column=text_column, label_column=label_column
        )
    if suffix == ".json":
        return _load_from_json_file(path, text_column=text_column, label_column=label_column)
    if suffix == ".jsonl":
        return _load_from_jsonl_file(path, text_column=text_column, label_column=label_column)

    raise ValueError(
        f"Formato no soportado para {path}. Usa CSV, TSV, JSON, JSONL o carpeta por clases."
    )


def sample_records(
    records: list[TextRecord],
    *,
    subset_size: int,
    seed: int,
) -> list[TextRecord]:
    if subset_size <= 0 or subset_size >= len(records):
        return list(records)
    rng = random.Random(seed)
    sampled = list(records)
    rng.shuffle(sampled)
    return sampled[:subset_size]


def stratified_split(
    records: list[TextRecord],
    *,
    val_size: float,
    seed: int,
) -> tuple[list[TextRecord], list[TextRecord]]:
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size debe estar entre 0 y 1")

    grouped: dict[str, list[TextRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)

    rng = random.Random(seed)
    train_records: list[TextRecord] = []
    val_records: list[TextRecord] = []

    for label in sorted(grouped):
        items = list(grouped[label])
        rng.shuffle(items)
        if len(items) == 1:
            train_records.extend(items)
            continue

        n_val = int(round(len(items) * val_size))
        n_val = max(1, min(n_val, len(items) - 1))
        val_records.extend(items[:n_val])
        train_records.extend(items[n_val:])

    if not train_records or not val_records:
        raise ValueError(
            "No se pudo crear una division train/val valida. "
            "Asegurate de tener suficientes ejemplos por clase."
        )

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def prepare_datasets(
    *,
    train_path: str | Path,
    vocab: Vocabulary,
    cfg,
    text_column: str = "text",
    label_column: str = "label",
    val_path: str | Path | None = None,
) -> tuple[TextDataset, TextDataset, LabelEncoder]:
    train_records = load_text_records(
        train_path,
        text_column=text_column,
        label_column=label_column,
    )
    train_records = sample_records(
        train_records,
        subset_size=cfg.subset_size,
        seed=cfg.seed,
    )

    if val_path:
        val_records = load_text_records(
            val_path,
            text_column=text_column,
            label_column=label_column,
        )
    else:
        train_records, val_records = stratified_split(
            train_records,
            val_size=cfg.val_size,
            seed=cfg.seed,
        )

    label_encoder = LabelEncoder.fit([record.label for record in train_records + val_records])
    train_dataset = TextDataset(train_records, vocab, label_encoder, max_len=cfg.max_len)
    val_dataset = TextDataset(val_records, vocab, label_encoder, max_len=cfg.max_len)
    return train_dataset, val_dataset, label_encoder


@torch.inference_mode()
def predict_texts(
    model,
    texts: list[str],
    vocab: Vocabulary,
    cfg,
    *,
    label_encoder: LabelEncoder | None = None,
) -> list[dict[str, Any]]:
    if not texts:
        return []

    model.eval()
    encoded = [vocab.encode(text, max_len=cfg.max_len) for text in texts]
    lengths = torch.tensor([max(1, len(ids)) for ids in encoded], dtype=torch.long)
    max_batch_len = int(lengths.max().item())
    batch = torch.full((len(encoded), max_batch_len), vocab.pad_idx, dtype=torch.long)
    for row, token_ids in enumerate(encoded):
        seq = torch.tensor(token_ids[:max_batch_len], dtype=torch.long)
        batch[row, : seq.numel()] = seq

    batch = batch.to(cfg.device)
    lengths = lengths.to(cfg.device)
    logits, _ = model(batch, lengths)
    probabilities = torch.softmax(logits, dim=-1).cpu()
    predictions = probabilities.argmax(dim=-1)

    outputs: list[dict[str, Any]] = []
    for text, probs, pred in zip(texts, probabilities, predictions):
        label_id = int(pred.item())
        label_name = label_encoder.decode(label_id) if label_encoder is not None else str(label_id)
        outputs.append(
            {
                "text": text,
                "label_id": label_id,
                "label_name": label_name,
                "confidence": float(probs[label_id].item()),
                "probabilities": probs.tolist(),
            }
        )
    return outputs
