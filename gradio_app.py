"""
gradio_app.py - Interfaz web opcional con Gradio.
"""

from __future__ import annotations

import csv
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from inferencia_avanzada import load_inference_bundle, predict_texts_detailed
from visualizacion import render_attention_html


TEXT_SUFFIXES = {".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".jsonl"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm", ".mp4"}


def _normalize_text(text: str | None) -> str:
    return str(text or "").strip()


def _resolve_uploaded_path(file_obj: Any) -> Path:
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj)
    for attr in ("path", "name", "filepath"):
        value = getattr(file_obj, attr, None)
        if value:
            return Path(value)
    raise ValueError("No se pudo resolver la ruta del archivo subido")


def _chunk_text(text: str, *, max_chars: int = 1200) -> list[str]:
    value = _normalize_text(text)
    if not value:
        return []

    paragraphs = [piece.strip() for piece in value.splitlines()]
    paragraphs = [piece for piece in paragraphs if piece]
    if not paragraphs:
        paragraphs = [value]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
            continue
        for start in range(0, len(paragraph), max_chars):
            chunks.append(paragraph[start : start + max_chars].strip())
        current = ""
    if current:
        chunks.append(current)
    return chunks


def _read_csv_like(path: Path, *, text_column: str) -> list[dict[str, Any]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"El archivo {path.name} no tiene encabezado")
        normalized = {name.strip().lower(): name for name in reader.fieldnames}
        key = text_column.strip().lower()
        if key not in normalized:
            raise ValueError(
                f"No existe la columna '{text_column}' en {path.name}. "
                f"Columnas disponibles: {', '.join(reader.fieldnames)}"
            )
        real_key = normalized[key]
        rows: list[dict[str, Any]] = []
        for row_idx, row in enumerate(reader, 1):
            value = _normalize_text(row.get(real_key))
            if not value:
                continue
            rows.append(
                {
                    "source": f"{path.name}#{row_idx}",
                    "text": value,
                    "origin": path.name,
                    "kind": "row",
                }
            )
        return rows


def _read_json_like(path: Path, *, text_column: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data", payload)
    if not isinstance(payload, list):
        raise ValueError(f"{path.name} debe contener una lista de objetos")

    rows: list[dict[str, Any]] = []
    for row_idx, item in enumerate(payload, 1):
        if not isinstance(item, dict):
            continue
        value = _normalize_text(item.get(text_column))
        if not value:
            continue
        rows.append(
            {
                "source": f"{path.name}#{row_idx}",
                "text": value,
                "origin": path.name,
                "kind": "row",
            }
        )
    return rows


def _load_text_units(path: Path, *, text_column: str = "text") -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return _read_csv_like(path, text_column=text_column)
    if suffix == ".json":
        return _read_json_like(path, text_column=text_column)
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for row_idx, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    continue
                value = _normalize_text(item.get(text_column))
                if not value:
                    continue
                rows.append(
                    {
                        "source": f"{path.name}#{row_idx}",
                        "text": value,
                        "origin": path.name,
                        "kind": "row",
                    }
                )
        return rows
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = _chunk_text(text)
    return [
        {
            "source": f"{path.name}#chunk_{idx}",
            "text": chunk,
            "origin": path.name,
            "kind": "chunk",
        }
        for idx, chunk in enumerate(chunks, 1)
    ]


@lru_cache(maxsize=1)
def _load_whisper_model(model_name: str):
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "Para analizar audio instala `openai-whisper` y ten disponible `ffmpeg` en el sistema."
        ) from exc
    return whisper.load_model(model_name)


def _transcribe_audio_file(path: Path, *, model_name: str = "base") -> str:
    model = _load_whisper_model(model_name)
    result = model.transcribe(str(path), fp16=False)
    return _normalize_text(result.get("text"))


def _analyze_texts(bundle, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    texts = [row["text"] for row in rows]
    predictions = predict_texts_detailed(bundle, texts)
    combined: list[dict[str, Any]] = []
    for row, prediction in zip(rows, predictions):
        combined.append(
            {
                "source": row["source"],
                "origin": row["origin"],
                "kind": row["kind"],
                "text": prediction["text"],
                "label": prediction["label_name"],
                "confidence": round(float(prediction["confidence"]), 6),
                "probabilities": prediction["probabilities"],
                "attention": prediction["attention"],
                "tokens": prediction["tokens"],
            }
        )

    summary: dict[str, Any] = {
        "items": len(combined),
        "predominant_label": None,
        "avg_confidence": 0.0,
        "sources": sorted({row["origin"] for row in combined}),
    }
    if combined:
        labels: dict[str, int] = {}
        confidence_total = 0.0
        for row in combined:
            labels[row["label"]] = labels.get(row["label"], 0) + 1
            confidence_total += float(row["confidence"])
        summary["predominant_label"] = max(labels.items(), key=lambda item: item[1])[0]
        summary["avg_confidence"] = round(confidence_total / len(combined), 6)
    return combined, summary


def _rows_from_uploaded_file(file_obj: Any, *, text_column: str) -> list[dict[str, Any]]:
    path = _resolve_uploaded_path(file_obj)
    suffix = path.suffix.lower()
    if suffix in AUDIO_SUFFIXES:
        transcript = _transcribe_audio_file(path)
        return (
            [
                {
                    "source": path.name,
                    "origin": path.name,
                    "kind": "audio",
                    "text": transcript,
                }
            ]
            if transcript
            else []
        )
    if suffix in TEXT_SUFFIXES:
        return _load_text_units(path, text_column=text_column)

    text = _normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
    if not text:
        return []
    return [
        {
            "source": path.name,
            "origin": path.name,
            "kind": "text",
            "text": text,
        }
    ]


def _build_attention_html(result: dict[str, Any], class_names: list[str]) -> str:
    return render_attention_html(
        result,
        title="Mapa de atencion",
        class_names=class_names,
    )


def _format_result_details(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "prediccion": result["label_name"],
        "confianza": round(float(result["confidence"]), 6),
        "tokens": result["tokens"],
        "attention": result["attention"],
        "probabilidades": [
            {"clase": idx, "valor": float(value)}
            for idx, value in enumerate(result["probabilities"])
        ],
    }


def _new_process_entry(action: str, status: str, detail: str) -> dict[str, str]:
    return {
        "hora": time.strftime("%H:%M:%S"),
        "accion": action,
        "estado": status,
        "detalle": detail,
    }


def _build_process_payload(
    entries: list[dict[str, str]] | None,
    *,
    current_action: str,
    status: str,
    detail: str,
) -> tuple[str, list[list[str]], dict[str, Any], list[dict[str, str]]]:
    history = list(entries or [])
    history.append(_new_process_entry(current_action, status, detail))
    history = history[-20:]
    table = [
        [item["hora"], item["accion"], item["estado"], item["detalle"]]
        for item in reversed(history)
    ]
    snapshot = {
        "accion_actual": current_action,
        "estado": status,
        "detalle": detail,
        "eventos": len(history),
    }
    status_text = (
        f"**Estado:** {status}\n\n"
        f"**Accion:** {current_action}\n\n"
        f"**Detalle:** {detail}"
    )
    return status_text, table, snapshot, history


def launch_gradio_app(
    cfg,
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio no esta instalado. Instala 'gradio' para usar la interfaz web."
        ) from exc

    bundle = load_inference_bundle(cfg)

    def infer_text(text: str, process_entries: list[dict[str, str]]):
        value = _normalize_text(text)
        if not value:
            process = _build_process_payload(
                process_entries,
                current_action="Analisis de texto",
                status="sin_datos",
                detail="No se recibio texto para analizar.",
            )
            return {}, "<div>Escribe una resena para ver la atencion.</div>", {}, *process
        result = predict_texts_detailed(bundle, [value])[0]
        label_scores = {
            class_name: float(prob)
            for class_name, prob in zip(bundle.label_encoder.class_names, result["probabilities"])
        }
        attention_html = _build_attention_html(result, bundle.label_encoder.class_names)
        details = _format_result_details(result)
        process = _build_process_payload(
            process_entries,
            current_action="Analisis de texto",
            status="completado",
            detail=f"Prediccion {result['label_name']} con confianza {float(result['confidence']) * 100:.1f}%.",
        )
        return label_scores, attention_html, details, *process

    def analyze_files(files, text_column: str):
        uploaded = list(files or [])
        if not uploaded:
            return [], [], {}, "<div>Sube archivos para comenzar el analisis.</div>", {}

        rows: list[dict[str, Any]] = []
        for file_obj in uploaded:
            rows.extend(_rows_from_uploaded_file(file_obj, text_column=text_column))

        if not rows:
            return [], [], {}, "<div>No se pudo extraer texto util de los archivos subidos.</div>", {}

        combined, summary = _analyze_texts(bundle, rows)
        table_rows = [
            [
                item["source"],
                item["kind"],
                item["label"],
                item["confidence"],
                item["text"][:280],
            ]
            for item in combined
        ]
        choices = [f"{idx + 1}. {item['source']} -> {item['label']}" for idx, item in enumerate(combined)]
        first = combined[0]
        return (
            table_rows,
            choices,
            summary,
            _build_attention_html(first, bundle.label_encoder.class_names),
            _format_result_details(first),
        )

    def analyze_selected_result(
        selection: str,
        stored_rows: list[dict[str, Any]],
        process_entries: list[dict[str, str]],
    ):
        if not stored_rows:
            process = _build_process_payload(
                process_entries,
                current_action="Detalle de archivos",
                status="sin_datos",
                detail="Todavia no hay resultados para inspeccionar.",
            )
            return "<div>Analiza archivos primero.</div>", {}, *process
        if not selection:
            first = stored_rows[0]
            process = _build_process_payload(
                process_entries,
                current_action="Detalle de archivos",
                status="completado",
                detail=f"Mostrando {first['source']} con etiqueta {first['label']}.",
            )
            return _build_attention_html(first, bundle.label_encoder.class_names), _format_result_details(first), *process
        try:
            index = int(str(selection).split(".", 1)[0]) - 1
        except ValueError:
            index = 0
        index = max(0, min(index, len(stored_rows) - 1))
        result = stored_rows[index]
        process = _build_process_payload(
            process_entries,
            current_action="Detalle de archivos",
            status="completado",
            detail=f"Mostrando {result['source']} con etiqueta {result['label']}.",
        )
        return _build_attention_html(result, bundle.label_encoder.class_names), _format_result_details(result), *process

    def analyze_audio(audio_path: str, process_entries: list[dict[str, str]], progress=gr.Progress()):
        if not audio_path:
            process = _build_process_payload(
                process_entries,
                current_action="Analisis de audio",
                status="sin_datos",
                detail="No se recibio ningun archivo de audio.",
            )
            return {}, "<div>Sube un audio para transcribirlo y analizarlo.</div>", {}, *process
        path = Path(audio_path)
        progress(0.25, desc="Transcribiendo audio")
        transcript = _transcribe_audio_file(path)
        if not transcript:
            process = _build_process_payload(
                process_entries,
                current_action="Analisis de audio",
                status="error",
                detail=f"No se pudo transcribir {path.name}.",
            )
            return {}, "<div>No se pudo transcribir el audio.</div>", {}, *process
        progress(0.75, desc="Clasificando texto transcrito")
        result = predict_texts_detailed(bundle, [transcript])[0]
        label_scores = {
            class_name: float(prob)
            for class_name, prob in zip(bundle.label_encoder.class_names, result["probabilities"])
        }
        attention_html = _build_attention_html(
            {
                **result,
                "text": transcript,
            },
            bundle.label_encoder.class_names,
        )
        details = _format_result_details(result)
        details["transcripcion"] = transcript
        details["archivo"] = path.name
        progress(1.0, desc="Listo")
        process = _build_process_payload(
            process_entries,
            current_action="Analisis de audio",
            status="completado",
            detail=f"Audio {path.name} clasificado como {result['label_name']}.",
        )
        return label_scores, attention_html, details, *process

    def _analyze_files_wrapper(
        files,
        text_column: str,
        process_entries: list[dict[str, str]],
        progress=gr.Progress(),
    ):
        uploaded = list(files or [])
        rows: list[dict[str, Any]] = []
        total = max(len(uploaded), 1)
        for idx, file_obj in enumerate(uploaded, 1):
            progress((idx - 1) / total, desc=f"Analizando {idx}/{total}")
            rows.extend(_rows_from_uploaded_file(file_obj, text_column=text_column))

        if not rows:
            process = _build_process_payload(
                process_entries,
                current_action="Analisis de archivos",
                status="sin_datos",
                detail="No se pudo extraer contenido util de los archivos subidos.",
            )
            return [], [], {}, "<div>No se pudo extraer texto util de los archivos subidos.</div>", {}, [], *process

        progress(0.85, desc="Clasificando contenidos")
        combined, summary = _analyze_texts(bundle, rows)
        if not combined:
            process = _build_process_payload(
                process_entries,
                current_action="Analisis de archivos",
                status="sin_datos",
                detail="No hubo fragmentos validos para clasificar.",
            )
            return [], [], summary, "<div>No se pudo extraer texto util de los archivos subidos.</div>", {}, [], *process

        table_rows = [
            [
                item["source"],
                item["kind"],
                item["label"],
                item["confidence"],
                item["text"][:280],
            ]
            for item in combined
        ]
        choices = [f"{idx + 1}. {item['source']} -> {item['label']}" for idx, item in enumerate(combined)]
        first = combined[0]
        progress(1.0, desc="Listo")
        process = _build_process_payload(
            process_entries,
            current_action="Analisis de archivos",
            status="completado",
            detail=f"Se analizaron {len(combined)} elementos de {len(uploaded)} archivo(s).",
        )
        return (
            table_rows,
            choices,
            summary,
            _build_attention_html(first, bundle.label_encoder.class_names),
            _format_result_details(first),
            combined,
            *process,
        )

    with gr.Blocks(title="Clasificador de sentimientos con analisis de archivos y audio") as demo:
        gr.Markdown("# Clasificador de sentimientos")
        gr.Markdown(
            "Analiza texto, archivos subidos y audio transcrito con una sola interfaz."
        )

        stored_rows = gr.State([])
        process_entries = gr.State([])

        with gr.Tab("Texto"):
            text_input = gr.Textbox(
                lines=5,
                label="Resena",
                placeholder="Escribe aqui una resena para clasificar su sentimiento...",
            )
            text_button = gr.Button("Analizar texto", variant="primary")
            text_label = gr.Label(label="Sentimiento")
            text_html = gr.HTML(label="Mapa de atencion")
            text_json = gr.JSON(label="Detalles")
            text_status = gr.Markdown()
            text_process_table = gr.Dataframe(
                headers=["hora", "accion", "estado", "detalle"],
                label="Proceso",
                interactive=False,
                wrap=True,
            )
            text_process_json = gr.JSON(label="Estado")
            text_button.click(
                infer_text,
                inputs=[text_input, process_entries],
                outputs=[text_label, text_html, text_json, text_status, text_process_table, text_process_json, process_entries],
            )
            text_input.submit(
                infer_text,
                inputs=[text_input, process_entries],
                outputs=[text_label, text_html, text_json, text_status, text_process_table, text_process_json, process_entries],
            )

        with gr.Tab("Archivos"):
            file_input = gr.Files(label="Sube archivos", file_count="multiple")
            file_text_column = gr.Textbox(value="text", label="Columna de texto")
            file_button = gr.Button("Analizar archivos", variant="primary")
            file_summary = gr.JSON(label="Resumen")
            file_table = gr.Dataframe(
                headers=["archivo", "tipo", "prediccion", "confianza", "texto"],
                label="Resultados",
                interactive=False,
                wrap=True,
            )
            file_selector = gr.Dropdown(label="Ver detalle de resultado", choices=[])
            file_html = gr.HTML(label="Mapa de atencion")
            file_json = gr.JSON(label="Detalles")
            file_status = gr.Markdown()
            file_process_table = gr.Dataframe(
                headers=["hora", "accion", "estado", "detalle"],
                label="Proceso",
                interactive=False,
                wrap=True,
            )
            file_process_json = gr.JSON(label="Estado")

            file_button.click(
                _analyze_files_wrapper,
                inputs=[file_input, file_text_column, process_entries],
                outputs=[
                    file_table,
                    file_selector,
                    file_summary,
                    file_html,
                    file_json,
                    stored_rows,
                    file_status,
                    file_process_table,
                    file_process_json,
                    process_entries,
                ],
            )
            file_selector.change(
                analyze_selected_result,
                inputs=[file_selector, stored_rows, process_entries],
                outputs=[file_html, file_json, file_status, file_process_table, file_process_json, process_entries],
            )

        with gr.Tab("Audio"):
            audio_input = gr.Audio(label="Audio", sources=["upload"], type="filepath")
            audio_button = gr.Button("Transcribir y analizar", variant="primary")
            audio_label = gr.Label(label="Sentimiento")
            audio_html = gr.HTML(label="Mapa de atencion")
            audio_json = gr.JSON(label="Detalles")
            audio_status = gr.Markdown()
            audio_process_table = gr.Dataframe(
                headers=["hora", "accion", "estado", "detalle"],
                label="Proceso",
                interactive=False,
                wrap=True,
            )
            audio_process_json = gr.JSON(label="Estado")
            audio_button.click(
                analyze_audio,
                inputs=[audio_input, process_entries],
                outputs=[
                    audio_label,
                    audio_html,
                    audio_json,
                    audio_status,
                    audio_process_table,
                    audio_process_json,
                    process_entries,
                ],
            )

        with gr.Tab("Proceso"):
            process_status = gr.Markdown("**Estado:** esperando\n\n**Accion:** ninguna\n\n**Detalle:** todavia no hay actividad.")
            process_table = gr.Dataframe(
                headers=["hora", "accion", "estado", "detalle"],
                value=[],
                label="Historial reciente",
                interactive=False,
                wrap=True,
            )
            process_json = gr.JSON(
                value={"accion_actual": "ninguna", "estado": "esperando", "detalle": "sin actividad", "eventos": 0},
                label="Snapshot",
            )

            sync_process_button = gr.Button("Actualizar vista", variant="secondary")

            def _sync_process(entries: list[dict[str, str]]):
                if not entries:
                    return (
                        "**Estado:** esperando\n\n**Accion:** ninguna\n\n**Detalle:** todavia no hay actividad.",
                        [],
                        {"accion_actual": "ninguna", "estado": "esperando", "detalle": "sin actividad", "eventos": 0},
                    )
                status_text, table, snapshot, _ = _build_process_payload(
                    entries[:-1],
                    current_action=entries[-1]["accion"],
                    status=entries[-1]["estado"],
                    detail=entries[-1]["detalle"],
                )
                return status_text, table, snapshot

            sync_process_button.click(
                _sync_process,
                inputs=process_entries,
                outputs=[process_status, process_table, process_json],
            )

    demo.queue()
    demo.launch(server_name=host, server_port=port, share=share)
