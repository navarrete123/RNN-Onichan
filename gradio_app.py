"""
gradio_app.py - Interfaz web opcional con Gradio.
"""

from __future__ import annotations

from inferencia_avanzada import load_inference_bundle, predict_texts_detailed
from visualizacion import render_attention_html


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

    def infer(text: str):
        value = str(text or "").strip()
        if not value:
            return {}, "<div>Escribe una resena para ver la atencion.</div>", {}
        result = predict_texts_detailed(bundle, [value])[0]
        label_scores = {
            class_name: float(prob)
            for class_name, prob in zip(bundle.label_encoder.class_names, result["probabilities"])
        }
        attention_html = render_attention_html(
            result,
            title="Atencion en tiempo real",
            class_names=bundle.label_encoder.class_names,
        )
        details = {
            "prediccion": result["label_name"],
            "confianza": round(float(result["confidence"]), 6),
            "tokens": result["tokens"],
            "attention": result["attention"],
        }
        return label_scores, attention_html, details

    demo = gr.Interface(
        fn=infer,
        inputs=gr.Textbox(
            lines=5,
            label="Resena",
            placeholder="Escribe aqui una resena para clasificar su sentimiento...",
        ),
        outputs=[
            gr.Label(label="Sentimiento"),
            gr.HTML(label="Mapa de atencion"),
            gr.JSON(label="Detalles"),
        ],
        title="Clasificador de sentimientos con atencion",
        description="Prediccion en tiempo real con visualizacion de atencion por token.",
        live=True,
    )
    demo.launch(server_name=host, server_port=port, share=share)
