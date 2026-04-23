"""
configuracion.py — Configuracion centralizada del proyecto RNN.
"""

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


@dataclass
class Config:
    # ── Modelo ────────────────────────────────────────────────────
    embed_dim:          int   = 160
    hidden_dim:         int   = 256
    num_layers:         int   = 2
    num_classes:        int   = 2
    dropout:            float = 0.35
    attention_dropout:  float = 0.10
    classifier_dim:     int   = 512
    rnn_type:           str   = "lstm"   # "lstm" | "gru"
    bidirectional:      bool  = True

    # ── Datos ─────────────────────────────────────────────────────
    max_vocab:   int   = 30_000
    min_freq:    int   = 2
    max_len:     int   = 256
    val_size:    float = 0.10
    # subset_size: N para entrenamiento rapido (0 = dataset completo)
    subset_size: int   = 0

    # ── Entrenamiento ─────────────────────────────────────────────
    epochs:                  int   = 10
    batch_size:              int   = 64
    lr:                      float = 1e-3
    weight_decay:            float = 1e-2
    label_smoothing:         float = 0.05
    grad_clip:               float = 1.0
    pct_warmup:              float = 0.10
    early_stopping_patience: int   = 3
    # log_interval: imprime progreso cada N batches (0 = desactivado)
    log_interval:            int   = 0
    show_progress:          bool  = True
    show_eval_progress:     bool  = False
    progress_refresh_steps: int   = 1

    # ── Optimizaciones del sistema ────────────────────────────────
    # compile_model: usa torch.compile() si PyTorch >= 2.0 (solo GPU, ~10-20% mas rapido)
    compile_model: bool = False

    # ── Sistema ───────────────────────────────────────────────────
    seed:            int  = 42
    num_workers:     int  = 0
    device:          str  = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    pin_memory:      bool = field(default_factory=lambda: torch.cuda.is_available())
    use_amp:         bool = field(default_factory=lambda: torch.cuda.is_available())
    checkpoint_path: str  = "mejor_modelo.pt"
    vocab_path:      str  = "vocab.json"

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

    # ── Propiedades calculadas ────────────────────────────────────
    @property
    def hidden_total(self) -> int:
        """Dimension de salida del RNN (x2 si bidireccional)."""
        return self.hidden_dim * (2 if self.bidirectional else 1)

    @property
    def feature_dim(self) -> int:
        """Vector final: atencion + mean pool + max pool + ultimo hidden."""
        return self.hidden_total * 4

    @property
    def device_type(self) -> str:
        return "cuda" if str(self.device).startswith("cuda") else "cpu"

    @property
    def is_quick_mode(self) -> bool:
        return self.subset_size > 0

    # ── Persistencia ─────────────────────────────────────────────
    def save(self, path: str | Path = "config.json"):
        """Guarda la config en JSON para reproducibilidad."""
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        print(f"Config guardada -> {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Carga una config desde JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    # ── Display ───────────────────────────────────────────────────
    def show(self):
        w = 26
        sep = "-" * (w + 22)
        print(sep)
        print("  Configuracion activa")
        print(sep)
        for key, value in self.__dict__.items():
            print(f"  {key:<{w}} {value}")
        print(f"  {'hidden_total':<{w}} {self.hidden_total}")
        print(f"  {'feature_dim':<{w}} {self.feature_dim}")
        if self.is_quick_mode:
            print(f"  {'!! MODO RAPIDO':<{w}} subset={self.subset_size} ejemplos")
        print(sep)


# ── Presets ───────────────────────────────────────────────────────
PRESETS: dict[str, dict] = {
    "clasificacion_texto": dict(
        embed_dim=160, hidden_dim=256, num_layers=2,
        dropout=0.35, attention_dropout=0.10, classifier_dim=512,
        rnn_type="lstm", bidirectional=True,
        max_vocab=30_000, min_freq=2, max_len=256,
        lr=1e-3, weight_decay=1e-2, label_smoothing=0.05,
        early_stopping_patience=3,
    ),
    # Mismo preset pero con subset de 2000 para probar rapido (~2 min/epoca en CPU)
    "clasificacion_rapida": dict(
        embed_dim=160, hidden_dim=256, num_layers=2,
        dropout=0.35, attention_dropout=0.10, classifier_dim=512,
        rnn_type="lstm", bidirectional=True,
        max_vocab=30_000, min_freq=2, max_len=256,
        lr=1e-3, weight_decay=1e-2, label_smoothing=0.05,
        early_stopping_patience=3, subset_size=2_000,
    ),
    "generacion_texto": dict(
        embed_dim=256, hidden_dim=512, num_layers=3,
        dropout=0.30, rnn_type="lstm", bidirectional=False,
    ),
    "series_temporales": dict(
        embed_dim=64, hidden_dim=128, num_layers=2,
        dropout=0.20, rnn_type="gru", bidirectional=False,
    ),
    "ner_pos_tagging": dict(
        embed_dim=192, hidden_dim=256, num_layers=2,
        dropout=0.40, rnn_type="lstm", bidirectional=True,
    ),
}


def get_config(preset: str | None = None, **overrides) -> Config:
    """
    Crea una Config desde un preset con overrides opcionales.

    Ejemplos:
        cfg = get_config("clasificacion_texto", epochs=20)
        cfg = get_config("clasificacion_rapida")   # subset 2000, rapido
        cfg = get_config(lr=5e-4, epochs=5)        # sin preset
    """
    base = PRESETS.get(preset, {}).copy() if preset else {}
    base.update(overrides)
    return Config(**base)


def seed_everything(seed: int):
    """Fija todas las semillas para reproducibilidad total."""
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
