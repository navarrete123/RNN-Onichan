import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from configuracion import Config, get_config, seed_everything
from datos_texto import (
    LabelEncoder,
    Vocabulary,
    build_text_loader,
    predict_texts,
    prepare_datasets,
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


class SyntheticTextDataset(Dataset):
    """Dataset sintetico para probar el pipeline de entrenamiento."""

    def __init__(self, num_samples: int, max_len: int, vocab_size: int, seed: int = 42):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        min_len = min(max(8, max_len // 8), max_len)

        self.lengths = torch.randint(
            low=min_len,
            high=max_len + 1,
            size=(num_samples,),
            generator=generator,
        )
        self.x = torch.zeros((num_samples, max_len), dtype=torch.long)
        self.y = torch.zeros(num_samples, dtype=torch.long)

        split_token = max(2, vocab_size // 2)
        for idx, length in enumerate(self.lengths.tolist()):
            label = torch.randint(0, 2, (1,), generator=generator).item()
            if label == 0:
                seq = torch.randint(1, split_token, (length,), generator=generator)
            else:
                seq = torch.randint(split_token, vocab_size, (length,), generator=generator)

            self.x[idx, :length] = seq
            self.y[idx] = label

    def __len__(self) -> int:
        return self.y.size(0)

    def __getitem__(self, idx: int):
        return self.x[idx], self.lengths[idx], self.y[idx]


def build_loader(dataset: Dataset, cfg, shuffle: bool) -> DataLoader:
    kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.num_workers,
    }
    if cfg.device_type == "cuda":
        kwargs["pin_memory"] = cfg.pin_memory
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def build_runtime_config(args) -> Config:
    overrides = {
        key: value
        for key, value in [
            ("epochs", args.epochs),
            ("lr", args.lr),
            ("subset_size", args.subset),
            ("batch_size", args.batch_size),
            ("max_len", args.max_len),
            ("num_workers", args.num_workers),
            ("seed", args.seed),
            ("log_interval", args.log_interval),
            ("progress_refresh_steps", args.progress_refresh_steps),
        ]
        if value is not None
    }
    overrides["compile_model"] = args.compile
    overrides["show_progress"] = not args.no_progress
    overrides["show_eval_progress"] = args.eval_progress
    if args.vocab_path:
        overrides["vocab_path"] = args.vocab_path
    if args.checkpoint_path:
        overrides["checkpoint_path"] = args.checkpoint_path
    return get_config(args.preset, **overrides)


def print_dataset_summary(train_dataset, val_dataset, vocab: Vocabulary, label_encoder: LabelEncoder):
    print("\nResumen de datos")
    print("----------------")
    print(f"  train samples : {len(train_dataset)}")
    print(f"  val samples   : {len(val_dataset)}")
    print(f"  vocab size    : {len(vocab)}")
    print(f"  clases        : {', '.join(label_encoder.class_names)}")


def modo_smoke_test(cfg: Config):
    vocab_size = min(cfg.max_vocab, 4_000)
    base_samples = cfg.subset_size or 1_024
    train_samples = max(base_samples, cfg.batch_size * 8)
    val_samples = max(train_samples // 4, cfg.batch_size * 2)

    print("\nEjecutando smoke test con dataset sintetico...")
    train_ds = SyntheticTextDataset(train_samples, cfg.max_len, vocab_size, seed=cfg.seed)
    val_ds = SyntheticTextDataset(val_samples, cfg.max_len, vocab_size, seed=cfg.seed + 1)
    train_loader = build_loader(train_ds, cfg, shuffle=True)
    val_loader = build_loader(val_ds, cfg, shuffle=False)

    model = MejorRNN(cfg, vocab_size=vocab_size)
    print(model)
    print(model.parameter_summary())

    label_state = {"class_names": ["0", "1"], "label_to_id": {"0": 0, "1": 1}}
    best_model = entrenar(model, train_loader, val_loader, cfg, label_state=label_state)
    val_loss, val_acc, val_f1 = evaluate(best_model, val_loader, cfg)
    print(
        f"\nSmoke test finalizado | val_loss={val_loss:.4f} "
        f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
    )


def modo_entrenar_texto(cfg: Config, args):
    vocab = Vocabulary.load(cfg.vocab_path)
    train_dataset, val_dataset, label_encoder = prepare_datasets(
        train_path=args.train_data,
        val_path=args.val_data,
        vocab=vocab,
        cfg=cfg,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    cfg.num_classes = len(label_encoder)
    cfg.show()
    print_dataset_summary(train_dataset, val_dataset, vocab, label_encoder)

    train_loader = build_text_loader(train_dataset, cfg, shuffle=True)
    val_loader = build_text_loader(val_dataset, cfg, shuffle=False)

    model = MejorRNN(cfg, vocab_size=len(vocab))
    print(model)
    print(model.parameter_summary())

    resume_checkpoint = None
    if args.resume:
        checkpoint_path = Path(cfg.checkpoint_path)
        if checkpoint_path.exists():
            resume_checkpoint = str(checkpoint_path)
        else:
            print(f"Advertencia: no existe checkpoint en {checkpoint_path}, se entrenara desde cero.")

    best_model = entrenar(
        model,
        train_loader,
        val_loader,
        cfg,
        vocab_state=vocab.state_dict(),
        label_state=label_encoder.state_dict(),
        resume_checkpoint=resume_checkpoint,
    )

    report = evaluate_detailed(
        best_model,
        val_loader,
        cfg,
        class_names=label_encoder.class_names,
    )
    print(
        f"\nValidacion final | loss={report['loss']:.4f} "
        f"acc={report['accuracy']:.4f} f1={report['f1']:.4f}"
    )
    print("\nReporte por clase")
    print_classification_report(report["confusion_matrix"], report["class_names"])
    print("\nMatriz de confusion")
    print_confusion_matrix(report["confusion_matrix"], report["class_names"])


def load_inference_bundle(cfg: Config):
    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe checkpoint en {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location="cpu")
    if "config" not in raw:
        raise ValueError("El checkpoint no contiene la configuracion del modelo")

    infer_cfg = Config(**raw["config"])
    infer_cfg.device = cfg.device
    infer_cfg.pin_memory = cfg.pin_memory
    infer_cfg.use_amp = cfg.use_amp
    infer_cfg.show_progress = cfg.show_progress
    infer_cfg.show_eval_progress = cfg.show_eval_progress
    infer_cfg.progress_refresh_steps = cfg.progress_refresh_steps

    if "vocab" in raw:
        vocab = Vocabulary.from_state(raw["vocab"])
    else:
        vocab_path = Path(cfg.vocab_path)
        if not vocab_path.exists():
            raise FileNotFoundError(
                "El checkpoint no trae vocabulario embebido y tampoco existe vocab.json"
            )
        vocab = Vocabulary.load(vocab_path)

    label_encoder = None
    if "label_encoder" in raw:
        label_encoder = LabelEncoder.from_state(raw["label_encoder"])
        infer_cfg.num_classes = len(label_encoder)

    model = MejorRNN(infer_cfg, vocab_size=len(vocab))
    load_checkpoint(str(checkpoint_path), model, map_location=infer_cfg.device)
    model = model.to(infer_cfg.device)
    return infer_cfg, model, vocab, label_encoder


def modo_infer_texto(cfg: Config, texts: list[str]):
    infer_cfg, model, vocab, label_encoder = load_inference_bundle(cfg)
    predictions = predict_texts(
        model,
        texts,
        vocab,
        infer_cfg,
        label_encoder=label_encoder,
    )
    print("\nInferencia")
    print("----------")
    for idx, item in enumerate(predictions, 1):
        print(f"[{idx}] {item['label_name']}  conf={item['confidence']:.4f}")
        print(f"    texto: {item['text']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="RNN para clasificacion de sentimiento con pipeline profesional"
    )
    parser.add_argument(
        "--preset",
        default="clasificacion_texto",
        choices=["clasificacion_texto", "clasificacion_rapida"],
        help="Preset de configuracion base",
    )
    parser.add_argument("--train-data", help="Ruta a CSV/TSV/JSON/JSONL o carpeta por clases")
    parser.add_argument("--val-data", help="Ruta opcional de validacion")
    parser.add_argument("--vocab-path", help="Ruta al vocabulario JSON")
    parser.add_argument("--checkpoint-path", help="Ruta del checkpoint a guardar/cargar")
    parser.add_argument("--text-column", default="text", help="Columna de texto para CSV/JSON")
    parser.add_argument("--label-column", default="label", help="Columna de etiqueta")
    parser.add_argument("--resume", action="store_true", help="Reanuda desde checkpoint")
    parser.add_argument(
        "--infer-text",
        nargs="+",
        help="Uno o mas textos para inferencia usando el checkpoint actual",
    )
    parser.add_argument("--epochs", type=int, help="Cantidad de epocas")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--subset", type=int, help="Subset rapido del dataset")
    parser.add_argument("--batch-size", type=int, dest="batch_size", help="Tamano de batch")
    parser.add_argument("--max-len", type=int, dest="max_len", help="Longitud maxima")
    parser.add_argument("--workers", type=int, dest="num_workers", help="Workers del DataLoader")
    parser.add_argument("--seed", type=int, help="Semilla global")
    parser.add_argument("--log-interval", type=int, help="Log cada N batches si no usas barra")
    parser.add_argument(
        "--progress-refresh",
        type=int,
        dest="progress_refresh_steps",
        help="Cada cuantos batches refrescar la barra",
    )
    parser.add_argument("--compile", action="store_true", help="Activa torch.compile si aplica")
    parser.add_argument("--no-progress", action="store_true", help="Desactiva la barra de progreso")
    parser.add_argument(
        "--eval-progress",
        action="store_true",
        help="Muestra barra tambien en validacion",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Entrena con un dataset sintetico para validar el pipeline",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        cfg = build_runtime_config(args)
    except Exception as exc:
        print(f"Error cargando configuracion: {exc}")
        sys.exit(1)

    seed_everything(cfg.seed)

    if args.smoke_test:
        cfg.show()
        modo_smoke_test(cfg)
        return

    if args.infer_text:
        modo_infer_texto(cfg, args.infer_text)
        return

    if args.train_data:
        modo_entrenar_texto(cfg, args)
        return

    cfg.show()
    raise RuntimeError(
        "Falta indicar que quieres hacer. Usa `--train-data <ruta>` para entrenar, "
        "`--infer-text \"tu texto\"` para inferir o `--smoke-test` para validar el pipeline."
    )


if __name__ == "__main__":
    main()
