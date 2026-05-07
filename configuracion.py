"""
configuracion.py - Configuracion centralizada del proyecto RNN.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


@dataclass
class Config:
    # Modelo
    embed_dim: int = 160
    hidden_dim: int = 256
    num_layers: int = 2
    num_classes: int = 2
    dropout: float = 0.35
    attention_dropout: float = 0.10
    classifier_dim: int = 512
    rnn_type: str = "lstm"
    bidirectional: bool = True

    # Datos
    max_vocab: int = 30_000
    min_freq: int = 2
    max_len: int = 256
    val_size: float = 0.10
    subset_size: int = 0

    # Entrenamiento
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    pct_warmup: float = 0.10
    early_stopping_patience: int = 3
    log_interval: int = 0
    show_progress: bool = True
    show_eval_progress: bool = False
    progress_refresh_steps: int = 1
    compile_model: bool = False

    # Mejoras opcionales
    embeddings_path: str | None = None
    freeze_embeddings: bool = False
    normalize_embeddings: bool = False
    augmentation_enabled: bool = False
    augmentation_synonyms_path: str | None = None
    augmentation_synonym_prob: float = 0.15
    augmentation_swap_prob: float = 0.10
    augmentation_delete_prob: float = 0.05
    augmentation_max_ops: int = 2
    ensemble_size: int = 1

    # Sistema
    seed: int = 42
    num_workers: int = 0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    pin_memory: bool = field(default_factory=lambda: torch.cuda.is_available())
    use_amp: bool = field(default_factory=lambda: torch.cuda.is_available())
    checkpoint_path: str = "mejor_modelo.pt"
    vocab_path: str = "vocab.json"
    artifacts_dir: str = "artifacts"

    # Tracking
    tracking_backend: str = "none"
    tracking_project: str = "rnn-sentimiento"
    tracking_run_name: str | None = None

    def __post_init__(self):
        if self.rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"rnn_type debe ser 'lstm' o 'gru', no '{self.rnn_type}'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout debe estar en [0, 1)")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout debe estar en [0, 1)")
        if not 0.0 < self.val_size < 0.5:
            raise ValueError("val_size debe estar en (0, 0.5)")
        if self.min_freq < 1:
            raise ValueError("min_freq debe ser >= 1")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience debe ser >= 1")
        if self.subset_size < 0:
            raise ValueError("subset_size debe ser >= 0")
        if self.progress_refresh_steps < 1:
            raise ValueError("progress_refresh_steps debe ser >= 1")
        if not 0.0 <= self.augmentation_synonym_prob <= 1.0:
            raise ValueError("augmentation_synonym_prob debe estar en [0, 1]")
        if not 0.0 <= self.augmentation_swap_prob <= 1.0:
            raise ValueError("augmentation_swap_prob debe estar en [0, 1]")
        if not 0.0 <= self.augmentation_delete_prob <= 1.0:
            raise ValueError("augmentation_delete_prob debe estar en [0, 1]")
        if self.augmentation_max_ops < 0:
            raise ValueError("augmentation_max_ops debe ser >= 0")
        if self.ensemble_size < 1:
            raise ValueError("ensemble_size debe ser >= 1")
        if self.tracking_backend not in {"none", "mlflow", "wandb"}:
            raise ValueError("tracking_backend debe ser 'none', 'mlflow' o 'wandb'")

    @property
    def hidden_total(self) -> int:
        return self.hidden_dim * (2 if self.bidirectional else 1)

    @property
    def feature_dim(self) -> int:
        return self.hidden_total * 4

    @property
    def device_type(self) -> str:
        return "cuda" if str(self.device).startswith("cuda") else "cpu"

    @property
    def is_quick_mode(self) -> bool:
        return self.subset_size > 0

    def save(self, path: str | Path = "config.json"):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        print(f"Config guardada -> {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def show(self):
        width = 28
        separator = "-" * (width + 24)
        print(separator)
        print("  Configuracion activa")
        print(separator)
        for key, value in self.__dict__.items():
            print(f"  {key:<{width}} {value}")
        print(f"  {'hidden_total':<{width}} {self.hidden_total}")
        print(f"  {'feature_dim':<{width}} {self.feature_dim}")
        if self.is_quick_mode:
            print(f"  {'modo_rapido':<{width}} subset={self.subset_size}")
        print(separator)


PRESETS: dict[str, dict] = {
    "clasificacion_texto": dict(
        embed_dim=160,
        hidden_dim=256,
        num_layers=2,
        dropout=0.35,
        attention_dropout=0.10,
        classifier_dim=512,
        rnn_type="lstm",
        bidirectional=True,
        max_vocab=30_000,
        min_freq=2,
        max_len=256,
        lr=1e-3,
        weight_decay=1e-2,
        label_smoothing=0.05,
        early_stopping_patience=3,
    ),
    "clasificacion_rapida": dict(
        embed_dim=160,
        hidden_dim=256,
        num_layers=2,
        dropout=0.35,
        attention_dropout=0.10,
        classifier_dim=512,
        rnn_type="lstm",
        bidirectional=True,
        max_vocab=30_000,
        min_freq=2,
        max_len=256,
        lr=1e-3,
        weight_decay=1e-2,
        label_smoothing=0.05,
        early_stopping_patience=3,
        subset_size=2_000,
    ),
    "generacion_texto": dict(
        embed_dim=256,
        hidden_dim=512,
        num_layers=3,
        dropout=0.30,
        rnn_type="lstm",
        bidirectional=False,
    ),
    "series_temporales": dict(
        embed_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.20,
        rnn_type="gru",
        bidirectional=False,
    ),
    "ner_pos_tagging": dict(
        embed_dim=192,
        hidden_dim=256,
        num_layers=2,
        dropout=0.40,
        rnn_type="lstm",
        bidirectional=True,
    ),
}


def get_config(preset: str | None = None, **overrides) -> Config:
    base = PRESETS.get(preset, {}).copy() if preset else {}
    base.update(overrides)
    return Config(**base)


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    cfg = get_config("clasificacion_texto")
    cfg.show()
    cfg.save("config.json")
    cfg2 = Config.load("config.json")
    assert cfg2.hidden_total == cfg.hidden_total
    print("Serializacion OK")
