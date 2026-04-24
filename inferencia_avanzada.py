"""
inferencia_avanzada.py - Utilidades de inferencia, ensemble, batch y ONNX.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from configuracion import Config
from datos_texto import LabelEncoder, Vocabulary
from entrenamiento import load_checkpoint
from modelo_profesional import MejorRNN


@dataclass
class InferenceBundle:
    cfg: Config
    model: MejorRNN
    vocab: Vocabulary
    label_encoder: LabelEncoder
    checkpoint_path: str


@dataclass
class EnsembleBundle:
    bundles: list[InferenceBundle]

    @property
    def class_names(self) -> list[str]:
        return self.bundles[0].label_encoder.class_names if self.bundles else []


def _prepare_batch(
    texts: list[str],
    vocab: Vocabulary,
    *,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor, list[list[str]], list[list[int]]]:
    token_lists = [vocab.tokenize(text)[:max_len] or [vocab.unk_token] for text in texts]
    id_lists = [vocab.encode_tokens(tokens, max_len=max_len) for tokens in token_lists]
    lengths = torch.tensor([max(1, len(ids)) for ids in id_lists], dtype=torch.long)
    max_batch_len = int(lengths.max().item())
    batch = torch.full((len(id_lists), max_batch_len), vocab.pad_idx, dtype=torch.long)
    for row, ids in enumerate(id_lists):
        seq = torch.tensor(ids[:max_batch_len], dtype=torch.long)
        batch[row, : seq.numel()] = seq
    return batch, lengths, token_lists, id_lists


def load_inference_bundle(
    cfg: Config,
    *,
    checkpoint_path: str | None = None,
) -> InferenceBundle:
    ckpt_path = Path(checkpoint_path or cfg.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No existe checkpoint en '{ckpt_path}'")

    raw = torch.load(str(ckpt_path), map_location="cpu")

    infer_cfg = Config(**raw["config"]) if "config" in raw else cfg
    infer_cfg.device = cfg.device
    infer_cfg.show_progress = getattr(cfg, "show_progress", False)
    infer_cfg.show_eval_progress = getattr(cfg, "show_eval_progress", False)

    if "vocab" in raw:
        vocab = Vocabulary.from_state(raw["vocab"])
    else:
        vocab = Vocabulary.load(cfg.vocab_path)

    if "label_encoder" in raw:
        label_encoder = LabelEncoder.from_state(raw["label_encoder"])
        infer_cfg.num_classes = len(label_encoder)
    else:
        label_encoder = LabelEncoder(["negativo", "positivo"])

    model = MejorRNN(infer_cfg, vocab_size=len(vocab))
    load_checkpoint(str(ckpt_path), model, map_location=infer_cfg.device)
    model = model.to(infer_cfg.device)
    model.eval()

    return InferenceBundle(
        cfg=infer_cfg,
        model=model,
        vocab=vocab,
        label_encoder=label_encoder,
        checkpoint_path=str(ckpt_path),
    )


@torch.inference_mode()
def predict_texts_detailed(bundle: InferenceBundle, texts: list[str]) -> list[dict[str, Any]]:
    if not texts:
        return []

    batch, lengths, token_lists, _ = _prepare_batch(
        texts,
        bundle.vocab,
        max_len=bundle.cfg.max_len,
    )
    logits, weights = bundle.model(
        batch.to(bundle.cfg.device),
        lengths.to(bundle.cfg.device),
    )
    probabilities = torch.softmax(logits, dim=-1).cpu()
    attention = weights.cpu()
    predictions = probabilities.argmax(dim=-1)

    outputs: list[dict[str, Any]] = []
    for idx, text in enumerate(texts):
        length = int(lengths[idx].item())
        pred_id = int(predictions[idx].item())
        probs = probabilities[idx].tolist()
        outputs.append(
            {
                "text": text,
                "tokens": token_lists[idx][:length],
                "attention": attention[idx, :length].tolist(),
                "label_id": pred_id,
                "label_name": bundle.label_encoder.decode(pred_id),
                "confidence": float(probabilities[idx, pred_id].item()),
                "probabilities": probs,
                "class_names": list(bundle.label_encoder.class_names),
            }
        )
    return outputs


def load_ensemble(checkpoints: list[str], cfg: Config) -> EnsembleBundle:
    bundles = [
        load_inference_bundle(cfg, checkpoint_path=checkpoint)
        for checkpoint in checkpoints
    ]
    if not bundles:
        raise ValueError("Se requiere al menos un checkpoint para ensemble")
    class_names = bundles[0].label_encoder.class_names
    for bundle in bundles[1:]:
        if bundle.label_encoder.class_names != class_names:
            raise ValueError("Todos los modelos del ensemble deben compartir las mismas clases")
    return EnsembleBundle(bundles=bundles)


@torch.inference_mode()
def predict_texts_ensemble(ensemble: EnsembleBundle, texts: list[str]) -> list[dict[str, Any]]:
    if not texts:
        return []
    if len(ensemble.bundles) == 1:
        return predict_texts_detailed(ensemble.bundles[0], texts)

    members = [predict_texts_detailed(bundle, texts) for bundle in ensemble.bundles]
    outputs: list[dict[str, Any]] = []

    for row_idx, text in enumerate(texts):
        per_model = [rows[row_idx] for rows in members]
        num_classes = len(per_model[0]["probabilities"])
        avg_probs = [0.0] * num_classes
        for result in per_model:
            for idx, value in enumerate(result["probabilities"]):
                avg_probs[idx] += float(value)
        avg_probs = [value / len(per_model) for value in avg_probs]

        pred_id = max(range(num_classes), key=lambda idx: avg_probs[idx])
        token_reference = per_model[0]["tokens"]
        attention_reference = per_model[0]["attention"]
        if all(len(item["attention"]) == len(attention_reference) for item in per_model):
            avg_attention = []
            for token_idx in range(len(attention_reference)):
                avg_attention.append(
                    sum(float(item["attention"][token_idx]) for item in per_model) / len(per_model)
                )
        else:
            avg_attention = attention_reference

        outputs.append(
            {
                "text": text,
                "tokens": token_reference,
                "attention": avg_attention,
                "label_id": pred_id,
                "label_name": ensemble.class_names[pred_id],
                "confidence": avg_probs[pred_id],
                "probabilities": avg_probs,
                "class_names": list(ensemble.class_names),
                "ensemble_size": len(per_model),
            }
        )
    return outputs


def _read_delimited_rows(path: Path) -> tuple[str, list[dict[str, str]], list[str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"El archivo {path} no tiene encabezado")
        rows = list(reader)
        return delimiter, rows, list(reader.fieldnames)


def save_batch_predictions(
    predictions: list[dict[str, Any]],
    input_path: str | Path,
    output_path: str | Path,
    *,
    text_column: str = "text",
) -> str:
    source = Path(input_path)
    destination = Path(output_path)
    delimiter, rows, fieldnames = _read_delimited_rows(source)
    normalized = {name.strip().lower(): name for name in fieldnames}
    key = text_column.strip().lower()
    if key not in normalized:
        raise ValueError(f"No existe la columna '{text_column}' en {source}")

    out_fields = list(fieldnames) + [
        "pred_label",
        "pred_label_id",
        "confidence",
        "probabilities_json",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields, delimiter=delimiter)
        writer.writeheader()
        for row, prediction in zip(rows, predictions):
            row = dict(row)
            row["pred_label"] = prediction["label_name"]
            row["pred_label_id"] = prediction["label_id"]
            row["confidence"] = f"{float(prediction['confidence']):.6f}"
            row["probabilities_json"] = json.dumps(prediction["probabilities"])
            writer.writerow(row)
    return str(destination)


def run_batch_prediction(
    bundle_or_ensemble: InferenceBundle | EnsembleBundle,
    input_path: str | Path,
    output_path: str | Path,
    *,
    text_column: str = "text",
) -> str:
    source = Path(input_path)
    _, rows, fieldnames = _read_delimited_rows(source)
    normalized = {name.strip().lower(): name for name in fieldnames}
    key = text_column.strip().lower()
    if key not in normalized:
        raise ValueError(f"No existe la columna '{text_column}' en {source}")

    texts = [str(row[normalized[key]]) for row in rows]
    if isinstance(bundle_or_ensemble, EnsembleBundle):
        predictions = predict_texts_ensemble(bundle_or_ensemble, texts)
    else:
        predictions = predict_texts_detailed(bundle_or_ensemble, texts)
    return save_batch_predictions(predictions, source, output_path, text_column=text_column)


class OnnxExportWrapper(nn.Module):
    def __init__(self, model: MejorRNN):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor):
        logits, attention = self.model.forward_exportable(tokens, lengths)
        probabilities = torch.softmax(logits, dim=-1)
        return logits, probabilities, attention


def export_bundle_to_onnx(
    bundle: InferenceBundle,
    output_path: str | Path,
    *,
    sample_text: str = "this movie was excellent",
    opset: int = 17,
) -> str:
    if not hasattr(bundle.model, "forward_exportable"):
        raise RuntimeError("El modelo no expone un camino exportable a ONNX")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    batch, lengths, _, _ = _prepare_batch(
        [sample_text],
        bundle.vocab,
        max_len=bundle.cfg.max_len,
    )

    wrapper = OnnxExportWrapper(bundle.model).to(bundle.cfg.device)
    wrapper.eval()

    try:
        torch.onnx.export(
            wrapper,
            (batch.to(bundle.cfg.device), lengths.to(bundle.cfg.device)),
            str(path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["tokens", "lengths"],
            output_names=["logits", "probabilities", "attention"],
            dynamic_axes={
                "tokens": {0: "batch_size", 1: "seq_len"},
                "lengths": {0: "batch_size"},
                "logits": {0: "batch_size"},
                "probabilities": {0: "batch_size"},
                "attention": {0: "batch_size", 1: "seq_len"},
            },
        )
    except Exception as exc:
        raise RuntimeError(
            "No se pudo exportar a ONNX. Instala 'onnx' y, si quieres validar el modelo, "
            "tambien 'onnxruntime'. Detalle: "
            f"{exc}"
        ) from exc
    return str(path)
