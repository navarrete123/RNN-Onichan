"""
main.py — Punto de entrada del proyecto RNN.

Modos de uso:
  python main.py                                      # entrena con IMDB (default)
  python main.py --preset clasificacion_rapida        # subset 2000, ~2 min/epoca
  python main.py --subset 3000 --epochs 5             # subset personalizado
  python main.py --resume                             # retoma desde checkpoint
  python main.py --test-only                          # evalua checkpoint en test set
  python main.py --infer-text "great movie!"          # prediccion texto libre
  python main.py --train-data datos.csv               # datos propios (CSV/JSON/carpeta)
  python main.py --smoke-test                         # prueba rapida con datos sinteticos
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from configuracion import Config, get_config, seed_everything
from datos_texto import (
    LabelEncoder,
    TextDataset,
    TextRecord,
    Vocabulary,
    build_text_loader,
    predict_texts,
    prepare_datasets,
    stratified_split,
    sample_records,
)
from entrenamiento import (
    entrenar,
    evaluate,
    evaluate_detailed,
    load_checkpoint,
    print_classification_report,
    print_confusion_matrix,
)
from modelo_profesional import MejorRNN


# ══════════════════════════════════════════════════════════════════
#  IMDB — modo por defecto
# ══════════════════════════════════════════════════════════════════

def _build_imdb_vocab(train_texts: list[str], cfg: Config) -> Vocabulary:
    """Construye vocabulario desde los textos de IMDB y lo guarda en disco."""
    vocab = Vocabulary.build_from_texts(
        train_texts,
        max_size  = cfg.max_vocab,
        min_freq  = cfg.min_freq,
        lowercase = True,
    )
    vocab.save(cfg.vocab_path)
    return vocab


def _imdb_to_records(texts: list[str], labels: list[int]) -> list[TextRecord]:
    label_map = {0: "negativo", 1: "positivo"}
    return [TextRecord(text=t, label=label_map[l]) for t, l in zip(texts, labels)]


def preparar_imdb(cfg: Config):
    """
    Descarga IMDB con HuggingFace, construye vocab y devuelve
    (train_ds, valid_ds, test_ds, vocab, label_encoder).
    """
    from datasets import load_dataset

    print("\nCargando IMDB...")
    imdb = load_dataset("imdb")

    # Split estratificado train/val
    try:
        splits = imdb["train"].train_test_split(
            test_size=cfg.val_size, seed=cfg.seed, stratify_by_column="label"
        )
    except (TypeError, ValueError):
        print("Aviso: split sin estratificar.")
        splits = imdb["train"].train_test_split(test_size=cfg.val_size, seed=cfg.seed)

    train_raw = splits["train"]
    valid_raw = splits["test"]
    test_raw  = imdb["test"]

    # Modo rapido: recortar
    if cfg.is_quick_mode:
        n     = cfg.subset_size
        n_val = max(int(n * cfg.val_size), 100)
        train_raw = train_raw.select(range(min(n, len(train_raw))))
        valid_raw = valid_raw.select(range(min(n_val, len(valid_raw))))
        test_raw  = test_raw.select(range(min(max(n, 500), len(test_raw))))
        print(f"[MODO RAPIDO] train={len(train_raw)} | valid={len(valid_raw)} | test={len(test_raw)}")

    # Vocabulario
    vocab = _build_imdb_vocab(train_raw["text"], cfg)

    # Convertir a TextRecord y TextDataset
    label_encoder = LabelEncoder(["negativo", "positivo"])

    def _to_dataset(raw) -> TextDataset:
        records = _imdb_to_records(raw["text"], raw["label"])
        return TextDataset(records, vocab, label_encoder, max_len=cfg.max_len)

    train_ds = _to_dataset(train_raw)
    valid_ds = _to_dataset(valid_raw)
    test_ds  = _to_dataset(test_raw)

    print(f"Splits -> train: {len(train_ds):,} | valid: {len(valid_ds):,} | test: {len(test_ds):,}")
    return train_ds, valid_ds, test_ds, vocab, label_encoder


def modo_imdb(cfg: Config, resume: bool, test_only: bool):
    """Entrena o evalua sobre IMDB."""
    seed_everything(cfg.seed)

    # Si solo evaluamos, cargar vocab del checkpoint
    if test_only:
        _modo_test_imdb(cfg)
        return

    train_ds, valid_ds, test_ds, vocab, label_encoder = preparar_imdb(cfg)
    cfg.num_classes = len(label_encoder)

    train_loader = build_text_loader(train_ds, cfg, shuffle=True)
    valid_loader = build_text_loader(valid_ds, cfg, shuffle=False)
    test_loader  = build_text_loader(test_ds,  cfg, shuffle=False)

    modelo = MejorRNN(cfg, vocab_size=len(vocab))
    print(modelo)
    print(modelo.parameter_summary())

    resume_ckpt = str(cfg.checkpoint_path) if resume and Path(cfg.checkpoint_path).exists() else None
    if resume and not resume_ckpt:
        print(f"Aviso: no se encontro checkpoint en '{cfg.checkpoint_path}', entrenando desde cero.")

    modelo = entrenar(
        modelo, train_loader, valid_loader, cfg,
        vocab_state       = vocab.state_dict(),
        label_state       = label_encoder.state_dict(),
        resume_checkpoint = resume_ckpt,
    )

    # Evaluacion final en test
    report = evaluate_detailed(modelo, test_loader, cfg, class_names=label_encoder.class_names)
    print(f"\nTest final -> loss={report['loss']:.4f} | acc={report['accuracy']:.4f} | f1={report['f1']:.4f}")
    print("\nReporte por clase:")
    print_classification_report(report["confusion_matrix"], report["class_names"])
    print("\nMatriz de confusion:")
    print_confusion_matrix(report["confusion_matrix"], report["class_names"])

    cfg.save("config.json")
    _demo_predicciones(modelo, vocab, label_encoder, cfg)


def _modo_test_imdb(cfg: Config):
    """Solo evalua el checkpoint guardado sobre el test set de IMDB."""
    ckpt_path = Path(cfg.checkpoint_path)
    if not ckpt_path.exists():
        print(f"No se encontro checkpoint en '{ckpt_path}'")
        return

    infer_cfg, modelo, vocab, label_encoder = _load_inference_bundle(cfg)

    from datasets import load_dataset
    test_raw  = load_dataset("imdb", split="test")
    label_map = {"negativo": 0, "positivo": 1}
    records   = _imdb_to_records(test_raw["text"], test_raw["label"])
    test_ds   = TextDataset(records, vocab, label_encoder, max_len=infer_cfg.max_len)
    test_loader = build_text_loader(test_ds, infer_cfg, shuffle=False)

    report = evaluate_detailed(modelo, test_loader, infer_cfg, class_names=label_encoder.class_names)
    print(f"\nTest -> loss={report['loss']:.4f} | acc={report['accuracy']:.4f} | f1={report['f1']:.4f}")
    print_classification_report(report["confusion_matrix"], report["class_names"])
    print_confusion_matrix(report["confusion_matrix"], report["class_names"])


# ══════════════════════════════════════════════════════════════════
#  DATOS PROPIOS
# ══════════════════════════════════════════════════════════════════

def modo_entrenar_texto(cfg: Config, args):
    """Entrena sobre un dataset propio (CSV / JSON / carpeta)."""
    seed_everything(cfg.seed)

    # Si hay vocab guardado lo carga, si no lo construye desde los datos
    vocab_path = Path(cfg.vocab_path)
    if vocab_path.exists():
        vocab = Vocabulary.load(vocab_path)
    else:
        from datos_texto import load_text_records
        print("Construyendo vocabulario desde datos de entrenamiento...")
        all_records = load_text_records(
            args.train_data,
            text_column=args.text_column,
            label_column=args.label_column,
        )
        vocab = Vocabulary.build_from_texts(
            [r.text for r in all_records],
            max_size=cfg.max_vocab,
            min_freq=cfg.min_freq,
        )
        vocab.save(cfg.vocab_path)

    train_dataset, val_dataset, label_encoder = prepare_datasets(
        train_path   = args.train_data,
        val_path     = getattr(args, "val_data", None),
        vocab        = vocab,
        cfg          = cfg,
        text_column  = args.text_column,
        label_column = args.label_column,
    )
    cfg.num_classes = len(label_encoder)
    cfg.show()

    print(f"\n  train: {len(train_dataset):,} | valid: {len(val_dataset):,} | "
          f"vocab: {len(vocab):,} | clases: {', '.join(label_encoder.class_names)}")

    train_loader = build_text_loader(train_dataset, cfg, shuffle=True)
    val_loader   = build_text_loader(val_dataset,   cfg, shuffle=False)

    modelo = MejorRNN(cfg, vocab_size=len(vocab))
    print(modelo)
    print(modelo.parameter_summary())

    resume_ckpt = str(cfg.checkpoint_path) if args.resume and Path(cfg.checkpoint_path).exists() else None

    best_model = entrenar(
        modelo, train_loader, val_loader, cfg,
        vocab_state       = vocab.state_dict(),
        label_state       = label_encoder.state_dict(),
        resume_checkpoint = resume_ckpt,
    )

    report = evaluate_detailed(best_model, val_loader, cfg, class_names=label_encoder.class_names)
    print(f"\nValidacion final -> loss={report['loss']:.4f} acc={report['accuracy']:.4f} f1={report['f1']:.4f}")
    print_classification_report(report["confusion_matrix"], report["class_names"])
    print_confusion_matrix(report["confusion_matrix"], report["class_names"])
    cfg.save("config.json")


# ══════════════════════════════════════════════════════════════════
#  INFERENCIA
# ══════════════════════════════════════════════════════════════════

def _load_inference_bundle(cfg: Config):
    """Carga modelo + vocab + label_encoder desde checkpoint."""
    ckpt_path = Path(cfg.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No existe checkpoint en '{ckpt_path}'")

    raw = torch.load(str(ckpt_path), map_location="cpu")

    # Usar config del checkpoint para reconstruir el modelo exacto
    infer_cfg = Config(**raw["config"]) if "config" in raw else cfg
    infer_cfg.device           = cfg.device
    infer_cfg.show_progress    = cfg.show_progress
    infer_cfg.show_eval_progress = getattr(cfg, "show_eval_progress", False)

    # Vocabulario
    if "vocab" in raw:
        vocab = Vocabulary.from_state(raw["vocab"])
    else:
        vocab_path = Path(cfg.vocab_path)
        if not vocab_path.exists():
            raise FileNotFoundError(
                "El checkpoint no incluye vocabulario y no se encontro vocab.json"
            )
        vocab = Vocabulary.load(vocab_path)

    # LabelEncoder
    if "label_encoder" in raw:
        label_encoder = LabelEncoder.from_state(raw["label_encoder"])
        infer_cfg.num_classes = len(label_encoder)
    else:
        label_encoder = LabelEncoder(["negativo", "positivo"])

    modelo = MejorRNN(infer_cfg, vocab_size=len(vocab))
    load_checkpoint(str(ckpt_path), modelo, map_location=infer_cfg.device)
    modelo = modelo.to(infer_cfg.device)
    return infer_cfg, modelo, vocab, label_encoder


def modo_infer(cfg: Config, texts: list[str]):
    infer_cfg, modelo, vocab, label_encoder = _load_inference_bundle(cfg)
    resultados = predict_texts(modelo, texts, vocab, infer_cfg, label_encoder=label_encoder)
    print("\nInferencia")
    print("-" * 60)
    for i, r in enumerate(resultados, 1):
        bar = "█" * int(r["confidence"] * 20)
        print(f"[{i}] {r['label_name'].upper():10s} {r['confidence']*100:5.1f}%  {bar}")
        print(f"     {r['text'][:80]}{'...' if len(r['text']) > 80 else ''}")


def _demo_predicciones(modelo: MejorRNN, vocab: Vocabulary, label_encoder: LabelEncoder, cfg: Config):
    ejemplos = [
        "This movie was absolutely fantastic! Great acting and story.",
        "Terrible film. Boring, predictable and a complete waste of time.",
        "It was okay, nothing special but not bad either.",
        "One of the best performances I have ever seen in my life.",
        "I fell asleep halfway through. Utterly disappointing.",
    ]
    resultados = predict_texts(modelo, ejemplos, vocab, cfg, label_encoder=label_encoder)
    print("\nPredicciones de ejemplo")
    print("-" * 60)
    for r in resultados:
        s = "+" if r["label_name"] == "positivo" else "-"
        bar = "█" * int(r["confidence"] * 20)
        print(f"[{s}] {r['label_name'].upper():10s} {r['confidence']*100:5.1f}%  {bar}")
        print(f"     {r['text'][:80]}")


# ══════════════════════════════════════════════════════════════════
#  SMOKE TEST
# ══════════════════════════════════════════════════════════════════

class _SyntheticDataset(Dataset):
    def __init__(self, n: int, max_len: int, vocab_size: int, seed: int = 42):
        gen      = torch.Generator().manual_seed(seed)
        min_len  = max(8, max_len // 8)
        self.lengths = torch.randint(min_len, max_len + 1, (n,), generator=gen)
        self.x = torch.zeros((n, max_len), dtype=torch.long)
        self.y = torch.zeros(n, dtype=torch.long)
        split  = max(2, vocab_size // 2)
        for i, l in enumerate(self.lengths.tolist()):
            lbl = int(torch.randint(0, 2, (1,), generator=gen).item())
            lo, hi = (1, split) if lbl == 0 else (split, vocab_size)
            self.x[i, :l] = torch.randint(lo, hi, (l,), generator=gen)
            self.y[i] = lbl

    def __len__(self): return self.y.size(0)
    def __getitem__(self, i): return self.x[i], self.lengths[i], self.y[i]


def modo_smoke_test(cfg: Config):
    vocab_size = min(cfg.max_vocab, 4_000)
    n_train    = max(cfg.subset_size or 1_024, cfg.batch_size * 8)
    n_val      = max(n_train // 4, cfg.batch_size * 2)
    print(f"\nSmoke test con dataset sintetico (train={n_train}, val={n_val})...")
    train_ds = _SyntheticDataset(n_train, cfg.max_len, vocab_size, cfg.seed)
    val_ds   = _SyntheticDataset(n_val,   cfg.max_len, vocab_size, cfg.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)
    modelo = MejorRNN(cfg, vocab_size=vocab_size)
    print(modelo)
    print(modelo.parameter_summary())
    label_state = {"class_names": ["0", "1"], "label_to_id": {"0": 0, "1": 1}}
    best = entrenar(modelo, train_loader, val_loader, cfg, label_state=label_state)
    val_loss, val_acc, val_f1 = evaluate(best, val_loader, cfg)
    print(f"\nSmoke test OK | val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")


# ══════════════════════════════════════════════════════════════════
#  ARGPARSE
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="RNN — clasificacion de texto",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Modo
    mode = p.add_argument_group("Modo de ejecucion")
    mode.add_argument("--train-data",  dest="train_data",  default=None,
                      help="CSV/JSON/JSONL/carpeta para entrenar con datos propios")
    mode.add_argument("--val-data",    dest="val_data",    default=None,
                      help="Datos de validacion opcionales")
    mode.add_argument("--resume",      action="store_true",
                      help="Retoma entrenamiento desde el ultimo checkpoint")
    mode.add_argument("--test-only",   action="store_true",
                      help="Solo evalua el checkpoint guardado (sin entrenar)")
    mode.add_argument("--infer-text",  dest="infer_text", nargs="+", default=None,
                      help="Texto/s a clasificar con el modelo guardado")
    mode.add_argument("--smoke-test",  action="store_true",
                      help="Prueba rapida con datos sinteticos")

    # Config
    cfg_g = p.add_argument_group("Configuracion")
    cfg_g.add_argument("--preset",       default="clasificacion_texto",
                       choices=list(["clasificacion_texto", "clasificacion_rapida",
                                     "generacion_texto", "series_temporales", "ner_pos_tagging"]))
    cfg_g.add_argument("--epochs",       type=int)
    cfg_g.add_argument("--lr",           type=float)
    cfg_g.add_argument("--subset",       type=int,  dest="subset",
                       help="Usar solo N ejemplos (0 = completo)")
    cfg_g.add_argument("--batch-size",   type=int,  dest="batch_size")
    cfg_g.add_argument("--dropout",      type=float)
    cfg_g.add_argument("--weight-decay", type=float, dest="weight_decay")
    cfg_g.add_argument("--label-smoothing", type=float, dest="label_smoothing")
    cfg_g.add_argument("--patience",     type=int,  dest="early_stopping_patience")
    cfg_g.add_argument("--seed",         type=int)

    # Datos
    data_g = p.add_argument_group("Datos")
    data_g.add_argument("--text-column",  default="text",  dest="text_column")
    data_g.add_argument("--label-column", default="label", dest="label_column")
    data_g.add_argument("--vocab-path",   dest="vocab_path",       default=None)
    data_g.add_argument("--checkpoint",   dest="checkpoint_path",  default=None)

    # Sistema
    sys_g = p.add_argument_group("Sistema")
    sys_g.add_argument("--compile",       action="store_true")
    sys_g.add_argument("--no-progress",   action="store_true")
    sys_g.add_argument("--workers",       type=int, dest="num_workers")

    return p.parse_args()


def _build_cfg(args) -> Config:
    overrides: dict[str, Any] = {}
    for src, dst in [
        ("epochs",                  "epochs"),
        ("lr",                      "lr"),
        ("subset",                  "subset_size"),
        ("batch_size",              "batch_size"),
        ("dropout",                 "dropout"),
        ("weight_decay",            "weight_decay"),
        ("label_smoothing",         "label_smoothing"),
        ("early_stopping_patience", "early_stopping_patience"),
        ("seed",                    "seed"),
        ("num_workers",             "num_workers"),
        ("vocab_path",              "vocab_path"),
        ("checkpoint_path",         "checkpoint_path"),
    ]:
        val = getattr(args, src, None)
        if val is not None:
            overrides[dst] = val

    if args.compile:
        overrides["compile_model"] = True
    if args.no_progress:
        overrides["show_progress"] = False

    return get_config(args.preset, **overrides)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    try:
        cfg = _build_cfg(args)
    except Exception as exc:
        print(f"Error en configuracion: {exc}")
        sys.exit(1)

    # ── Despacho por modo ─────────────────────────────────────────
    if args.smoke_test:
        cfg.show()
        modo_smoke_test(cfg)

    elif args.infer_text:
        modo_infer(cfg, args.infer_text)

    elif args.train_data:
        modo_entrenar_texto(cfg, args)

    else:
        # Modo por defecto: IMDB
        cfg.show()
        modo_imdb(cfg, resume=args.resume, test_only=args.test_only)


if __name__ == "__main__":
    main()
