"""
main.py - Punto de entrada del proyecto RNN.

Ejemplos:
  python main.py
  python main.py --preset clasificacion_rapida
  python main.py --train-data datos.csv --augment
  python main.py --infer-text "great movie" --attention-html
  python main.py --batch-input reviews.csv --batch-output predicciones.csv
  python main.py --export-onnx
  python main.py --launch-gradio
  python main.py --serve-api
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from augmentacion_texto import TextAugmenter, load_synonym_map
from configuracion import Config, get_config, seed_everything
from datos_texto import (
    LabelEncoder,
    TextDataset,
    TextRecord,
    Vocabulary,
    build_text_loader,
    prepare_datasets,
)
from embeddings_utils import load_embeddings_into_model
from entrenamiento import (
    collect_prediction_rows,
    entrenar,
    evaluate,
    evaluate_detailed,
    highest_confidence_errors,
    print_classification_report,
    print_confusion_matrix,
)
from inferencia_avanzada import (
    EnsembleBundle,
    InferenceBundle,
    export_bundle_to_onnx,
    load_ensemble,
    load_inference_bundle,
    predict_texts_detailed,
    predict_texts_ensemble,
    run_batch_prediction,
)
from modelo_profesional import MejorRNN
from tracking_experimentos import create_tracker
from visualizacion import save_attention_html, save_error_analysis_html


def _artifact_dir(cfg: Config) -> Path:
    path = Path(cfg.artifacts_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_artifact_path(cfg: Config, filename: str) -> str:
    return str(_artifact_dir(cfg) / filename)


def _build_text_augmenter(cfg: Config, *, seed: int | None = None):
    if not cfg.augmentation_enabled:
        return None
    synonyms = load_synonym_map(cfg.augmentation_synonyms_path)
    return TextAugmenter(
        enabled=True,
        synonym_prob=cfg.augmentation_synonym_prob,
        swap_prob=cfg.augmentation_swap_prob,
        delete_prob=cfg.augmentation_delete_prob,
        max_ops=cfg.augmentation_max_ops,
        synonyms=synonyms,
        seed=cfg.seed if seed is None else seed,
    )


def _apply_model_enhancements(modelo: MejorRNN, vocab: Vocabulary, cfg: Config, tracker=None):
    if not cfg.embeddings_path:
        return
    stats = load_embeddings_into_model(
        modelo,
        vocab,
        embeddings_path=cfg.embeddings_path,
        normalize=cfg.normalize_embeddings,
        freeze=cfg.freeze_embeddings,
    )
    print(
        "Embeddings preentrenados cargados -> "
        f"tokens={stats['loaded_tokens']} cobertura={float(stats['coverage']):.2%}"
    )
    if tracker is not None:
        tracker.log_summary(
            {
                "embedding_loaded_tokens": float(stats["loaded_tokens"]),
                "embedding_coverage": float(stats["coverage"]),
                "embedding_detected_dim": float(stats["detected_dim"]),
            }
        )


def _make_tracker(cfg: Config, *, run_suffix: str | None = None):
    run_name = cfg.tracking_run_name or Path(cfg.checkpoint_path).stem
    if run_suffix:
        run_name = f"{run_name}{run_suffix}"
    tracker = create_tracker(
        backend=cfg.tracking_backend,
        project=cfg.tracking_project,
        run_name=run_name,
    )
    tracker.log_config(cfg)
    return tracker


def _train_single_model(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab: Vocabulary,
    label_encoder: LabelEncoder,
    *,
    resume: bool = False,
    run_suffix: str | None = None,
):
    tracker = _make_tracker(cfg, run_suffix=run_suffix)
    try:
        modelo = MejorRNN(cfg, vocab_size=len(vocab))
        _apply_model_enhancements(modelo, vocab, cfg, tracker=tracker)
        print(modelo)
        print(modelo.parameter_summary())

        resume_ckpt = str(cfg.checkpoint_path) if resume and Path(cfg.checkpoint_path).exists() else None
        if resume and not resume_ckpt:
            print(f"Aviso: no se encontro checkpoint en '{cfg.checkpoint_path}', entrenando desde cero.")

        modelo = entrenar(
            modelo,
            train_loader,
            val_loader,
            cfg,
            vocab_state=vocab.state_dict(),
            label_state=label_encoder.state_dict(),
            resume_checkpoint=resume_ckpt,
            tracker=tracker,
        )
        return modelo, tracker
    except Exception:
        tracker.finish("failed")
        raise


def _print_report(title: str, report: dict):
    print(
        f"\n{title} -> loss={report['loss']:.4f} "
        f"acc={report['accuracy']:.4f} f1={report['f1']:.4f}"
    )
    print("\nReporte por clase:")
    print_classification_report(report["confusion_matrix"], report["class_names"])
    print("\nMatriz de confusion:")
    print_confusion_matrix(report["confusion_matrix"], report["class_names"])


def _save_error_report(
    modelo: MejorRNN,
    loader: DataLoader,
    cfg: Config,
    class_names: list[str],
    *,
    destination: str | None,
    top_n: int,
):
    rows = collect_prediction_rows(modelo, loader, cfg, class_names=class_names)
    errors = highest_confidence_errors(rows, top_n=top_n)
    output = destination or _default_artifact_path(cfg, "error_analysis.html")
    path = save_error_analysis_html(errors, output)
    print(f"Analisis de errores guardado -> {path}")
    return path


def _bundle_from_model(
    cfg: Config,
    modelo: MejorRNN,
    vocab: Vocabulary,
    label_encoder: LabelEncoder,
) -> InferenceBundle:
    return InferenceBundle(
        cfg=cfg,
        model=modelo,
        vocab=vocab,
        label_encoder=label_encoder,
        checkpoint_path=cfg.checkpoint_path,
    )


def _load_predictor(cfg: Config, args):
    if args.ensemble_checkpoints:
        return load_ensemble(args.ensemble_checkpoints, cfg)
    return load_inference_bundle(cfg)


def _predict_with_loaded_target(target, texts: list[str]) -> list[dict[str, Any]]:
    if isinstance(target, EnsembleBundle):
        return predict_texts_ensemble(target, texts)
    return predict_texts_detailed(target, texts)


def _save_attention_if_requested(results: list[dict[str, Any]], cfg: Config, output_path: str | None):
    if not results or output_path is None:
        return None
    destination = output_path or _default_artifact_path(cfg, "attention.html")
    path = save_attention_html(
        results[0],
        destination,
        title="Visualizador de atencion",
        class_names=results[0].get("class_names"),
    )
    print(f"Visualizador de atencion guardado -> {path}")
    return path


def _build_imdb_vocab(train_texts: list[str], cfg: Config) -> Vocabulary:
    vocab = Vocabulary.build_from_texts(
        train_texts,
        max_size=cfg.max_vocab,
        min_freq=cfg.min_freq,
        lowercase=True,
    )
    vocab.save(cfg.vocab_path)
    return vocab


def _imdb_to_records(texts: list[str], labels: list[int]) -> list[TextRecord]:
    label_map = {0: "negativo", 1: "positivo"}
    return [TextRecord(text=text, label=label_map[label]) for text, label in zip(texts, labels)]


def preparar_imdb(cfg: Config):
    from datasets import load_dataset

    print("\nCargando IMDB...")
    imdb = load_dataset("imdb")

    try:
        splits = imdb["train"].train_test_split(
            test_size=cfg.val_size,
            seed=cfg.seed,
            stratify_by_column="label",
        )
    except (TypeError, ValueError):
        print("Aviso: split sin estratificar.")
        splits = imdb["train"].train_test_split(test_size=cfg.val_size, seed=cfg.seed)

    train_raw = splits["train"]
    valid_raw = splits["test"]
    test_raw = imdb["test"]

    if cfg.is_quick_mode:
        n = cfg.subset_size
        n_val = max(int(n * cfg.val_size), 100)
        train_raw = train_raw.select(range(min(n, len(train_raw))))
        valid_raw = valid_raw.select(range(min(n_val, len(valid_raw))))
        test_raw = test_raw.select(range(min(max(n, 500), len(test_raw))))
        print(f"[MODO RAPIDO] train={len(train_raw)} | valid={len(valid_raw)} | test={len(test_raw)}")

    vocab = _build_imdb_vocab(train_raw["text"], cfg)
    label_encoder = LabelEncoder(["negativo", "positivo"])
    train_augmenter = _build_text_augmenter(cfg)

    train_ds = TextDataset(
        _imdb_to_records(train_raw["text"], train_raw["label"]),
        vocab,
        label_encoder,
        max_len=cfg.max_len,
        augmenter=train_augmenter,
    )
    valid_ds = TextDataset(
        _imdb_to_records(valid_raw["text"], valid_raw["label"]),
        vocab,
        label_encoder,
        max_len=cfg.max_len,
    )
    test_ds = TextDataset(
        _imdb_to_records(test_raw["text"], test_raw["label"]),
        vocab,
        label_encoder,
        max_len=cfg.max_len,
    )

    print(f"Splits -> train: {len(train_ds):,} | valid: {len(valid_ds):,} | test: {len(test_ds):,}")
    return train_ds, valid_ds, test_ds, vocab, label_encoder


def _train_ensemble(
    cfg: Config,
    train_ds: TextDataset,
    valid_ds: TextDataset,
    vocab: Vocabulary,
    label_encoder: LabelEncoder,
    *,
    resume: bool,
) -> list[str]:
    ensemble_dir = _artifact_dir(cfg) / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: list[str] = []

    for idx in range(cfg.ensemble_size):
        run_cfg = copy.deepcopy(cfg)
        run_cfg.seed = cfg.seed + idx
        run_cfg.checkpoint_path = str(ensemble_dir / f"model_seed_{run_cfg.seed}.pt")
        run_cfg.artifacts_dir = str(ensemble_dir / f"seed_{run_cfg.seed}")
        seed_everything(run_cfg.seed)

        run_train_ds = TextDataset(
            train_ds.records,
            vocab,
            label_encoder,
            max_len=run_cfg.max_len,
            augmenter=_build_text_augmenter(run_cfg, seed=run_cfg.seed),
        )
        run_valid_ds = TextDataset(
            valid_ds.records,
            vocab,
            label_encoder,
            max_len=run_cfg.max_len,
        )
        train_loader = build_text_loader(run_train_ds, run_cfg, shuffle=True)
        valid_loader = build_text_loader(run_valid_ds, run_cfg, shuffle=False)

        print(f"\n[Ensemble {idx + 1}/{cfg.ensemble_size}] seed={run_cfg.seed}")
        modelo, tracker = _train_single_model(
            run_cfg,
            train_loader,
            valid_loader,
            vocab,
            label_encoder,
            resume=resume,
            run_suffix=f"-seed-{run_cfg.seed}",
        )
        report = evaluate_detailed(modelo, valid_loader, run_cfg, class_names=label_encoder.class_names)
        _print_report(f"Validacion modelo seed={run_cfg.seed}", report)
        tracker.log_summary(
            {
                "final_val_acc": report["accuracy"],
                "final_val_f1": report["f1"],
                "final_val_loss": report["loss"],
            }
        )
        tracker.finish("completed")
        checkpoints.append(run_cfg.checkpoint_path)

    manifest = {
        "ensemble_size": cfg.ensemble_size,
        "checkpoints": checkpoints,
        "class_names": label_encoder.class_names,
    }
    manifest_path = ensemble_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nEnsemble guardado -> {manifest_path}")
    return checkpoints


def modo_entrenar_texto(cfg: Config, args):
    seed_everything(cfg.seed)
    _artifact_dir(cfg)

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
            [record.text for record in all_records],
            max_size=cfg.max_vocab,
            min_freq=cfg.min_freq,
        )
        vocab.save(cfg.vocab_path)

    train_dataset, val_dataset, label_encoder = prepare_datasets(
        train_path=args.train_data,
        val_path=args.val_data,
        vocab=vocab,
        cfg=cfg,
        text_column=args.text_column,
        label_column=args.label_column,
        train_augmenter=_build_text_augmenter(cfg),
    )
    cfg.num_classes = len(label_encoder)
    cfg.show()

    print(
        f"\ntrain: {len(train_dataset):,} | valid: {len(val_dataset):,} | "
        f"vocab: {len(vocab):,} | clases: {', '.join(label_encoder.class_names)}"
    )

    if cfg.ensemble_size > 1:
        _train_ensemble(cfg, train_dataset, val_dataset, vocab, label_encoder, resume=args.resume)
        cfg.save(_artifact_dir(cfg) / "config.json")
        return

    train_loader = build_text_loader(train_dataset, cfg, shuffle=True)
    val_loader = build_text_loader(val_dataset, cfg, shuffle=False)

    modelo, tracker = _train_single_model(
        cfg,
        train_loader,
        val_loader,
        vocab,
        label_encoder,
        resume=args.resume,
    )
    report = evaluate_detailed(modelo, val_loader, cfg, class_names=label_encoder.class_names)
    _print_report("Validacion final", report)

    error_path = _save_error_report(
        modelo,
        val_loader,
        cfg,
        label_encoder.class_names,
        destination=args.error_analysis_html,
        top_n=args.error_top_n,
    )
    tracker.log_artifact(error_path)
    tracker.finish("completed")

    cfg.save(_artifact_dir(cfg) / "config.json")


def _modo_test_imdb(cfg: Config, args):
    infer_bundle = load_inference_bundle(cfg)
    from datasets import load_dataset

    test_raw = load_dataset("imdb", split="test")
    label_encoder = infer_bundle.label_encoder
    records = _imdb_to_records(test_raw["text"], test_raw["label"])
    test_ds = TextDataset(records, infer_bundle.vocab, label_encoder, max_len=infer_bundle.cfg.max_len)
    test_loader = build_text_loader(test_ds, infer_bundle.cfg, shuffle=False)

    report = evaluate_detailed(
        infer_bundle.model,
        test_loader,
        infer_bundle.cfg,
        class_names=label_encoder.class_names,
    )
    _print_report("Test", report)
    _save_error_report(
        infer_bundle.model,
        test_loader,
        infer_bundle.cfg,
        label_encoder.class_names,
        destination=args.error_analysis_html,
        top_n=args.error_top_n,
    )


def modo_imdb(cfg: Config, args):
    seed_everything(cfg.seed)
    _artifact_dir(cfg)

    if args.test_only:
        _modo_test_imdb(cfg, args)
        return

    train_ds, valid_ds, test_ds, vocab, label_encoder = preparar_imdb(cfg)
    cfg.num_classes = len(label_encoder)

    if cfg.ensemble_size > 1:
        _train_ensemble(cfg, train_ds, valid_ds, vocab, label_encoder, resume=args.resume)
        cfg.save(_artifact_dir(cfg) / "config.json")
        return

    train_loader = build_text_loader(train_ds, cfg, shuffle=True)
    valid_loader = build_text_loader(valid_ds, cfg, shuffle=False)
    test_loader = build_text_loader(test_ds, cfg, shuffle=False)

    modelo, tracker = _train_single_model(
        cfg,
        train_loader,
        valid_loader,
        vocab,
        label_encoder,
        resume=args.resume,
    )
    report = evaluate_detailed(modelo, test_loader, cfg, class_names=label_encoder.class_names)
    _print_report("Test final", report)

    error_path = _save_error_report(
        modelo,
        test_loader,
        cfg,
        label_encoder.class_names,
        destination=args.error_analysis_html,
        top_n=args.error_top_n,
    )
    tracker.log_artifact(error_path)
    tracker.finish("completed")

    cfg.save(_artifact_dir(cfg) / "config.json")
    _demo_predicciones(modelo, vocab, label_encoder, cfg, attention_html=args.attention_html)


def modo_infer(cfg: Config, args):
    target = _load_predictor(cfg, args)
    results = _predict_with_loaded_target(target, args.infer_text)
    print("\nInferencia")
    print("-" * 60)
    for idx, result in enumerate(results, 1):
        bar = "#" * max(1, int(float(result["confidence"]) * 20))
        print(f"[{idx}] {result['label_name'].upper():10s} {float(result['confidence']) * 100:5.1f}%  {bar}")
        print(f"     {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}")
    if args.attention_html is not None:
        _save_attention_if_requested(
            results,
            cfg,
            args.attention_html or _default_artifact_path(cfg, "attention.html"),
        )


def modo_batch(cfg: Config, args):
    target = _load_predictor(cfg, args)
    input_path = Path(args.batch_input)
    if args.batch_output:
        output_path = Path(args.batch_output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_predicciones{input_path.suffix}")
    path = run_batch_prediction(
        target,
        input_path,
        output_path,
        text_column=args.batch_text_column,
    )
    print(f"Predicciones batch guardadas -> {path}")


def modo_export_onnx(cfg: Config, args):
    if args.ensemble_checkpoints:
        raise ValueError("La exportacion a ONNX usa un solo checkpoint, no un ensemble.")
    bundle = load_inference_bundle(cfg)
    output_path = args.export_onnx
    if output_path is None or output_path == "model.onnx":
        output_path = _default_artifact_path(cfg, "model.onnx")
    path = export_bundle_to_onnx(
        bundle,
        output_path,
        sample_text=args.onnx_sample_text,
        opset=args.onnx_opset,
    )
    print(f"Modelo exportado a ONNX -> {path}")


def _demo_predicciones(
    modelo: MejorRNN,
    vocab: Vocabulary,
    label_encoder: LabelEncoder,
    cfg: Config,
    *,
    attention_html: str | None = None,
):
    ejemplos = [
        "This movie was absolutely fantastic. Great acting and story.",
        "Terrible film. Boring, predictable and a complete waste of time.",
        "It was okay, nothing special but not bad either.",
        "One of the best performances I have ever seen in my life.",
        "I fell asleep halfway through. Utterly disappointing.",
    ]
    bundle = _bundle_from_model(cfg, modelo, vocab, label_encoder)
    resultados = predict_texts_detailed(bundle, ejemplos)
    print("\nPredicciones de ejemplo")
    print("-" * 60)
    for result in resultados:
        sign = "+" if result["label_name"] == "positivo" else "-"
        bar = "#" * max(1, int(float(result["confidence"]) * 20))
        print(f"[{sign}] {result['label_name'].upper():10s} {float(result['confidence']) * 100:5.1f}%  {bar}")
        print(f"     {result['text'][:100]}")
    if attention_html is not None:
        _save_attention_if_requested(
            resultados,
            cfg,
            attention_html or _default_artifact_path(cfg, "attention_demo.html"),
        )


class _SyntheticDataset(Dataset):
    def __init__(self, n: int, max_len: int, vocab_size: int, seed: int = 42):
        gen = torch.Generator().manual_seed(seed)
        min_len = max(8, max_len // 8)
        self.lengths = torch.randint(min_len, max_len + 1, (n,), generator=gen)
        self.x = torch.zeros((n, max_len), dtype=torch.long)
        self.y = torch.zeros(n, dtype=torch.long)
        split = max(2, vocab_size // 2)
        for idx, length in enumerate(self.lengths.tolist()):
            label = int(torch.randint(0, 2, (1,), generator=gen).item())
            lo, hi = (1, split) if label == 0 else (split, vocab_size)
            self.x[idx, :length] = torch.randint(lo, hi, (length,), generator=gen)
            self.y[idx] = label

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.lengths[idx], self.y[idx]


def modo_smoke_test(cfg: Config):
    _artifact_dir(cfg)
    vocab_size = min(cfg.max_vocab, 4_000)
    n_train = max(cfg.subset_size or 1_024, cfg.batch_size * 8)
    n_val = max(n_train // 4, cfg.batch_size * 2)
    print(f"\nSmoke test con dataset sintetico (train={n_train}, val={n_val})...")
    train_ds = _SyntheticDataset(n_train, cfg.max_len, vocab_size, cfg.seed)
    val_ds = _SyntheticDataset(n_val, cfg.max_len, vocab_size, cfg.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    modelo = MejorRNN(cfg, vocab_size=vocab_size)
    print(modelo)
    print(modelo.parameter_summary())
    label_state = {"class_names": ["0", "1"], "label_to_id": {"0": 0, "1": 1}}
    vocab_state = {
        "word2idx": {
            "<PAD>": 0,
            "<UNK>": 1,
            **{f"tok_{idx}": idx for idx in range(2, vocab_size)},
        }
    }
    tracker = _make_tracker(cfg, run_suffix="-smoke")
    modelo = entrenar(
        modelo,
        train_loader,
        val_loader,
        cfg,
        vocab_state=vocab_state,
        label_state=label_state,
        tracker=tracker,
    )
    val_loss, val_acc, val_f1 = evaluate(modelo, val_loader, cfg)
    print(f"\nSmoke test OK | val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
    tracker.finish("completed")


def parse_args():
    parser = argparse.ArgumentParser(
        description="RNN - clasificacion de texto con analisis, exportacion y serving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_argument_group("Modo de ejecucion")
    mode.add_argument("--train-data", dest="train_data", default=None, help="CSV/JSON/JSONL/carpeta para entrenar")
    mode.add_argument("--val-data", dest="val_data", default=None, help="Datos de validacion opcionales")
    mode.add_argument("--resume", action="store_true", help="Retoma entrenamiento desde el ultimo checkpoint")
    mode.add_argument("--test-only", action="store_true", help="Evalua el checkpoint guardado sobre test")
    mode.add_argument("--infer-text", dest="infer_text", nargs="+", default=None, help="Texto/s a clasificar")
    mode.add_argument("--batch-input", dest="batch_input", default=None, help="CSV/TSV para prediccion batch")
    mode.add_argument("--batch-output", dest="batch_output", default=None, help="Ruta de salida batch")
    mode.add_argument("--export-onnx", nargs="?", const="model.onnx", default=None, help="Exporta el modelo a ONNX")
    mode.add_argument("--launch-gradio", action="store_true", help="Lanza una interfaz web con Gradio")
    mode.add_argument("--serve-api", action="store_true", help="Levanta una API REST con FastAPI")
    mode.add_argument("--smoke-test", action="store_true", help="Prueba rapida con datos sinteticos")

    cfg_group = parser.add_argument_group("Configuracion")
    cfg_group.add_argument(
        "--preset",
        default="clasificacion_texto",
        choices=["clasificacion_texto", "clasificacion_rapida", "generacion_texto", "series_temporales", "ner_pos_tagging"],
    )
    cfg_group.add_argument("--epochs", type=int)
    cfg_group.add_argument("--lr", type=float)
    cfg_group.add_argument("--subset", type=int, dest="subset")
    cfg_group.add_argument("--batch-size", type=int, dest="batch_size")
    cfg_group.add_argument("--dropout", type=float)
    cfg_group.add_argument("--weight-decay", type=float, dest="weight_decay")
    cfg_group.add_argument("--label-smoothing", type=float, dest="label_smoothing")
    cfg_group.add_argument("--patience", type=int, dest="early_stopping_patience")
    cfg_group.add_argument("--seed", type=int)
    cfg_group.add_argument("--artifacts-dir", dest="artifacts_dir", default=None)

    data_group = parser.add_argument_group("Datos")
    data_group.add_argument("--text-column", default="text", dest="text_column")
    data_group.add_argument("--label-column", default="label", dest="label_column")
    data_group.add_argument("--batch-text-column", default="text", dest="batch_text_column")
    data_group.add_argument("--vocab-path", dest="vocab_path", default=None)
    data_group.add_argument("--checkpoint", dest="checkpoint_path", default=None)

    model_group = parser.add_argument_group("Mejoras del modelo")
    model_group.add_argument("--embeddings-path", dest="embeddings_path", default=None, help="Archivo GloVe/FastText local")
    model_group.add_argument("--freeze-embeddings", action="store_true", help="Congela embeddings preentrenados")
    model_group.add_argument("--normalize-embeddings", action="store_true", help="Normaliza embeddings al cargarlos")
    model_group.add_argument("--augment", action="store_true", help="Activa data augmentation")
    model_group.add_argument("--augmentation-synonyms", dest="augmentation_synonyms_path", default=None, help="JSON opcional de sinonimos")
    model_group.add_argument("--augmentation-synonym-prob", dest="augmentation_synonym_prob", type=float)
    model_group.add_argument("--augmentation-swap-prob", dest="augmentation_swap_prob", type=float)
    model_group.add_argument("--augmentation-delete-prob", dest="augmentation_delete_prob", type=float)
    model_group.add_argument("--augmentation-max-ops", dest="augmentation_max_ops", type=int)
    model_group.add_argument("--ensemble-size", dest="ensemble_size", type=int)
    model_group.add_argument("--ensemble-checkpoints", nargs="+", default=None, help="Lista de checkpoints para promediar predicciones")

    analysis_group = parser.add_argument_group("Analisis y debugging")
    analysis_group.add_argument("--attention-html", nargs="?", const="", default=None, help="Guarda un HTML con pesos de atencion")
    analysis_group.add_argument("--error-analysis-html", dest="error_analysis_html", default=None, help="Ruta HTML para errores de alta confianza")
    analysis_group.add_argument("--error-top-n", dest="error_top_n", type=int, default=25)

    product_group = parser.add_argument_group("Productizacion")
    product_group.add_argument("--gradio-host", default="127.0.0.1")
    product_group.add_argument("--gradio-port", type=int, default=7860)
    product_group.add_argument("--gradio-share", action="store_true")
    product_group.add_argument("--api-host", default="127.0.0.1")
    product_group.add_argument("--api-port", type=int, default=8000)
    product_group.add_argument("--tracking", dest="tracking_backend", choices=["none", "mlflow", "wandb"], default=None)
    product_group.add_argument("--tracking-project", dest="tracking_project", default=None)
    product_group.add_argument("--tracking-run-name", dest="tracking_run_name", default=None)
    product_group.add_argument("--onnx-sample-text", dest="onnx_sample_text", default="this movie was excellent")
    product_group.add_argument("--onnx-opset", dest="onnx_opset", type=int, default=17)

    sys_group = parser.add_argument_group("Sistema")
    sys_group.add_argument("--compile", action="store_true")
    sys_group.add_argument("--no-progress", action="store_true")
    sys_group.add_argument("--workers", type=int, dest="num_workers")

    return parser.parse_args()


def _build_cfg(args) -> Config:
    overrides: dict[str, Any] = {}
    for src, dst in [
        ("epochs", "epochs"),
        ("lr", "lr"),
        ("subset", "subset_size"),
        ("batch_size", "batch_size"),
        ("dropout", "dropout"),
        ("weight_decay", "weight_decay"),
        ("label_smoothing", "label_smoothing"),
        ("early_stopping_patience", "early_stopping_patience"),
        ("seed", "seed"),
        ("num_workers", "num_workers"),
        ("vocab_path", "vocab_path"),
        ("checkpoint_path", "checkpoint_path"),
        ("artifacts_dir", "artifacts_dir"),
        ("embeddings_path", "embeddings_path"),
        ("augmentation_synonyms_path", "augmentation_synonyms_path"),
        ("augmentation_synonym_prob", "augmentation_synonym_prob"),
        ("augmentation_swap_prob", "augmentation_swap_prob"),
        ("augmentation_delete_prob", "augmentation_delete_prob"),
        ("augmentation_max_ops", "augmentation_max_ops"),
        ("ensemble_size", "ensemble_size"),
        ("tracking_backend", "tracking_backend"),
        ("tracking_project", "tracking_project"),
        ("tracking_run_name", "tracking_run_name"),
    ]:
        value = getattr(args, src, None)
        if value is not None:
            overrides[dst] = value

    if args.compile:
        overrides["compile_model"] = True
    if args.no_progress:
        overrides["show_progress"] = False
    if args.freeze_embeddings:
        overrides["freeze_embeddings"] = True
    if args.normalize_embeddings:
        overrides["normalize_embeddings"] = True
    if args.augment:
        overrides["augmentation_enabled"] = True

    return get_config(args.preset, **overrides)


def main():
    args = parse_args()

    try:
        cfg = _build_cfg(args)
    except Exception as exc:
        print(f"Error en configuracion: {exc}")
        sys.exit(1)

    try:
        if args.smoke_test:
            cfg.show()
            modo_smoke_test(cfg)

        elif args.launch_gradio:
            from gradio_app import launch_gradio_app

            launch_gradio_app(
                cfg,
                host=args.gradio_host,
                port=args.gradio_port,
                share=args.gradio_share,
            )

        elif args.serve_api:
            from api_fastapi import run_api

            run_api(cfg, host=args.api_host, port=args.api_port)

        elif args.export_onnx is not None:
            modo_export_onnx(cfg, args)

        elif args.batch_input:
            modo_batch(cfg, args)

        elif args.infer_text:
            modo_infer(cfg, args)

        elif args.train_data:
            modo_entrenar_texto(cfg, args)

        else:
            cfg.show()
            modo_imdb(cfg, args)

    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
