from datetime import datetime
from pathlib import Path

def generar_informe_detallado(cfg, history, test_report, nombre_archivo="informe_final.md"):
    """Analiza los resultados del entrenamiento y genera conclusiones en Markdown."""
    
    mejor_val_acc = max(h["val_acc"] for h in history)
    tr_final_acc = history[-1]["tr_acc"]
    val_final_acc = history[-1]["val_acc"]
    
    # Análisis de Overfitting
    gap = tr_final_acc - val_final_acc
    overfitting_status = "⚠️ Detectado" if gap > 0.1 else "✅ Bajo"

    informe = f"""# Informe de Resultados - RNN Onichan
## 1. Configuración del Experimento
- **Arquitectura:** {cfg.rnn_type.upper()} {'Bidireccional' if cfg.bidirectional else 'Unidireccional'}
- **Capas:** {cfg.num_layers} | **Hidden Dim:** {cfg.hidden_dim}
- **Regularización:** Dropout {cfg.dropout}, Weight Decay {cfg.weight_decay}

## 2. Métricas de Entrenamiento
- **Épocas ejecutadas:** {len(history)}
- **Mejor Precisión (Val):** {mejor_val_acc:.4f}
- **Brecha de Generalización:** {gap:.4f} ({overfitting_status})

## 3. Desempeño en Test (Final)
- **Precisión Total:** {test_report['accuracy']:.4f}
- **F1-Score Promedio:** {test_report['f1']:.4f}
- **Pérdida (Loss):** {test_report['loss']:.4f}

## 4. Conclusiones
"""
    if gap > 0.1:
        informe += "- El modelo muestra signos de sobreajuste. Se sugiere aumentar el `attention_dropout` o reducir `hidden_dim`.\n"
    if test_report['accuracy'] > 0.85:
        informe += "- Rendimiento excelente. La combinación de pooling de atención y global es efectiva para este dataset.\n"

    Path(nombre_archivo).write_text(informe, encoding="utf-8")
    print(f"Informe generado en {nombre_archivo}")


def generar_pdf_profesional(cfg, history, report, nombre_archivo="Informe_RNN_Onichan.pdf"):
    """
    Genera un informe PDF con diseno institucional, metricas y analisis automatico.
    """
    ahora = datetime.now().strftime("%d/%m/%Y %H:%M")

    gap = history[-1]["tr_acc"] - history[-1]["val_acc"]
    status_msg = "El modelo presenta una excelente capacidad de generalizacion."
    if gap > 0.1:
        status_msg = "Se detecta un posible sobreajuste (overfitting). Se recomienda aumentar el Dropout."
    elif history[-1]["val_acc"] < 0.7:
        status_msg = "El modelo tiene un rendimiento moderado. Considere aumentar el tamano de la capa oculta."

    class_rows = "".join(
        "<tr>"
        f"<td>{c['class_name']}</td>"
        f"<td>{c['precision']:.4f}</td>"
        f"<td>{c['recall']:.4f}</td>"
        f"<td>{c['f1']:.4f}</td>"
        "</tr>"
        for c in report["classwise"]
    )

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{ size: A4; margin: 20mm; }}
            body {{ font-family: Helvetica, sans-serif; color: #333; line-height: 1.6; }}
            .header {{ border-bottom: 2px solid #2563eb; padding-bottom: 10px; margin-bottom: 20px; }}
            h1 {{ color: #1e3a8a; margin: 0; font-size: 24pt; }}
            .summary-grid {{ display: flex; justify-content: space-between; margin: 20px 0; }}
            .card {{ background: #f1f5f9; padding: 15px; border-radius: 8px; text-align: center; width: 30%; }}
            .card-val {{ font-size: 18pt; font-weight: bold; color: #2563eb; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th {{ background: #e2e8f0; padding: 10px; text-align: left; }}
            td {{ padding: 10px; border-bottom: 1px solid #e2e8f0; }}
            .obs {{ background: #eff6ff; border-left: 4px solid #3b82f6; padding: 15px; font-style: italic; }}
            footer {{ position: fixed; bottom: 0; font-size: 9pt; color: #94a3b8; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Informe de Red Neuronal RNN-Onichan</h1>
            <p>Generado el {ahora} | Configuracion: {cfg.rnn_type.upper()}</p>
        </div>

        <div class="summary-grid">
            <div class="card">
                <div class="card-val">{report['accuracy'] * 100:.2f}%</div>
                <div class="stat-label">Accuracy en Test</div>
            </div>
            <div class="card">
                <div class="card-val">{len(history)}</div>
                <div class="stat-label">Epocas</div>
            </div>
            <div class="card">
                <div class="card-val">{cfg.hidden_dim}</div>
                <div class="stat-label">Dimension Oculta</div>
            </div>
        </div>

        <h2>Analisis de Resultados por Clase</h2>
        <table>
            <thead>
                <tr>
                    <th>Categoria</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                </tr>
            </thead>
            <tbody>
                {class_rows}
            </tbody>
        </table>

        <h2>Observaciones del Sistema</h2>
        <div class="obs">
            {status_msg} <br><br>
            <strong>Detalles tecnicos:</strong> El entrenamiento finalizo con una perdida de
            {report['loss']:.4f} y un F1-Score promedio de {report['f1']:.4f}.
        </div>

        <footer>
            RNN-Onichan Deep Learning Framework - Propiedad del Proyecto
        </footer>
    </body>
    </html>
    """

    html_fallback = Path(nombre_archivo).with_suffix(".html")

    try:
        from weasyprint import HTML
    except Exception as exc:
        html_fallback.write_text(html_template, encoding="utf-8")
        raise RuntimeError(
            "No se pudo cargar WeasyPrint para renderizar el PDF. "
            f"Se guardo una version HTML en {html_fallback}. "
            "Si el paquete no esta instalado, usa: pip install weasyprint. "
            "En Windows tambien puede faltar la dependencia nativa Pango/GTK."
        ) from exc

    try:
        HTML(string=html_template).write_pdf(nombre_archivo)
    except Exception as exc:
        html_fallback.write_text(html_template, encoding="utf-8")
        raise RuntimeError(
            "No se pudo renderizar el PDF con WeasyPrint. "
            f"Se guardo una version HTML en {html_fallback}. "
            "En Windows suele faltar la dependencia nativa Pango/GTK."
        ) from exc

    print(f"Informe PDF profesional generado: {nombre_archivo}")
