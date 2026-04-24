"""
visualizacion.py - HTMLs para atencion, errores y curvas de aprendizaje.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from uuid import uuid4


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _format_probabilities(probabilities: list[float], class_names: list[str] | None) -> str:
    names = class_names or [f"Clase {idx}" for idx in range(len(probabilities))]
    bars: list[str] = []
    for idx, value in enumerate(probabilities):
        label = html.escape(names[idx] if idx < len(names) else f"Clase {idx}")
        width = _clamp01(value) * 100.0
        bars.append(
            "<div class='prob-row'>"
            f"<div class='prob-label'>{label}</div>"
            f"<div class='prob-track'><div class='prob-fill' style='width:{width:.2f}%'></div></div>"
            f"<div class='prob-value'>{value * 100:.1f}%</div>"
            "</div>"
        )
    return "".join(bars)


def render_attention_html(
    prediction: dict,
    *,
    title: str = "Mapa de atencion",
    class_names: list[str] | None = None,
) -> str:
    element_id = f"attn-{uuid4().hex}"
    tokens = prediction.get("tokens") or []
    weights = prediction.get("attention") or []
    rows: list[str] = []
    if tokens and weights:
        max_weight = max(max(weights), 1e-8)
        for token, weight in zip(tokens, weights):
            norm = _clamp01(weight / max_weight)
            rows.append(
                "<span class='token' "
                f"data-weight='{weight:.6f}' data-norm='{norm:.6f}' "
                f"title='peso={weight:.4f}'>"
                f"{html.escape(str(token))}</span>"
            )
    else:
        rows.append("<span class='empty'>No hay pesos de atencion disponibles.</span>")

    probabilities = prediction.get("probabilities") or []
    probability_html = _format_probabilities(probabilities, class_names)
    label = html.escape(str(prediction.get("label_name", "-")))
    confidence = float(prediction.get("confidence", 0.0)) * 100.0
    source_text = html.escape(str(prediction.get("text", "")))

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: linear-gradient(135deg, #f7f4ea 0%, #f2f7fb 100%);
      color: #18324a;
    }}
    .card {{
      max-width: 980px;
      margin: 0 auto;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid rgba(24, 50, 74, 0.10);
      border-radius: 20px;
      box-shadow: 0 18px 60px rgba(24, 50, 74, 0.10);
      overflow: hidden;
    }}
    .hero {{
      padding: 22px 26px 14px 26px;
      background: linear-gradient(120deg, #dbe9f3 0%, #f9eddc 100%);
      border-bottom: 1px solid rgba(24, 50, 74, 0.08);
    }}
    .hero h1 {{
      margin: 0 0 6px 0;
      font-size: 28px;
    }}
    .meta {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      font-size: 14px;
      color: #4d657a;
    }}
    .body {{
      padding: 24px 26px 30px 26px;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 14px;
      margin-bottom: 18px;
      flex-wrap: wrap;
      font-size: 14px;
    }}
    .controls input {{
      width: 220px;
    }}
    .tokens {{
      line-height: 2.25;
      font-size: 20px;
      margin-bottom: 28px;
    }}
    .token {{
      display: inline-block;
      margin: 0 6px 8px 0;
      padding: 2px 8px;
      border-radius: 10px;
      background: rgba(233, 128, 36, 0.08);
      box-shadow: inset 0 0 0 1px rgba(233, 128, 36, 0.05);
      transition: transform 120ms ease, background 120ms ease;
    }}
    .token:hover {{
      transform: translateY(-1px);
    }}
    .section-title {{
      margin: 0 0 12px 0;
      font-size: 17px;
    }}
    .probabilities {{
      display: grid;
      gap: 10px;
      margin-bottom: 24px;
    }}
    .prob-row {{
      display: grid;
      grid-template-columns: 120px 1fr 72px;
      gap: 12px;
      align-items: center;
      font-size: 14px;
    }}
    .prob-track {{
      height: 10px;
      border-radius: 999px;
      background: rgba(24, 50, 74, 0.08);
      overflow: hidden;
    }}
    .prob-fill {{
      height: 100%;
      background: linear-gradient(90deg, #1f7a8c 0%, #f08700 100%);
      border-radius: inherit;
    }}
    .text-box {{
      margin-top: 12px;
      padding: 16px;
      border-radius: 14px;
      background: rgba(24, 50, 74, 0.04);
      font-size: 15px;
      line-height: 1.6;
      white-space: pre-wrap;
    }}
    .empty {{
      color: #6b7f91;
    }}
  </style>
</head>
<body>
  <div class="card" id="{element_id}">
    <div class="hero">
      <h1>{html.escape(title)}</h1>
      <div class="meta">
        <div><strong>Prediccion:</strong> {label}</div>
        <div><strong>Confianza:</strong> {confidence:.1f}%</div>
        <div><strong>Tokens:</strong> {len(tokens)}</div>
      </div>
    </div>
    <div class="body">
      <div class="controls">
        <label for="{element_id}-slider"><strong>Contraste de color</strong></label>
        <input id="{element_id}-slider" type="range" min="0.40" max="2.80" step="0.05" value="1.30">
        <span id="{element_id}-value">1.30x</span>
      </div>

      <div class="section-title">Texto con pesos de atencion</div>
      <div class="tokens">{''.join(rows)}</div>

      <div class="section-title">Distribucion de probabilidades</div>
      <div class="probabilities">{probability_html}</div>

      <div class="section-title">Texto original</div>
      <div class="text-box">{source_text}</div>
    </div>
  </div>
  <script>
    (function() {{
      const root = document.getElementById("{element_id}");
      const slider = document.getElementById("{element_id}-slider");
      const sliderValue = document.getElementById("{element_id}-value");
      const tokens = Array.from(root.querySelectorAll(".token"));

      function paint() {{
        const gain = Number(slider.value);
        sliderValue.textContent = gain.toFixed(2) + "x";
        for (const token of tokens) {{
          const weight = Number(token.dataset.norm || "0");
          const alpha = Math.min(0.95, Math.max(0.06, weight * gain));
          token.style.background = "rgba(240, 135, 0, " + alpha.toFixed(3) + ")";
          token.style.boxShadow = "inset 0 0 0 1px rgba(24, 50, 74, " + Math.min(0.25, alpha).toFixed(3) + ")";
        }}
      }}

      slider.addEventListener("input", paint);
      paint();
    }})();
  </script>
</body>
</html>"""


def save_attention_html(
    prediction: dict,
    output_path: str | Path,
    *,
    title: str = "Mapa de atencion",
    class_names: list[str] | None = None,
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_attention_html(prediction, title=title, class_names=class_names),
        encoding="utf-8",
    )
    return str(path)


def render_error_analysis_html(errors: list[dict], *, title: str = "Analisis de errores") -> str:
    rows: list[str] = []
    for idx, item in enumerate(errors, 1):
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{html.escape(str(item.get('true_label_name', '-')))}</td>"
            f"<td>{html.escape(str(item.get('pred_label_name', '-')))}</td>"
            f"<td>{float(item.get('confidence', 0.0)) * 100:.1f}%</td>"
            f"<td>{html.escape(str(item.get('text', '')))}</td>"
            "</tr>"
        )
    table_rows = "".join(rows) or (
        "<tr><td colspan='5'>No se encontraron errores para mostrar.</td></tr>"
    )
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: #f6f7fb;
      color: #1f2937;
    }}
    .wrap {{
      max-width: 1100px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 8px 0;
    }}
    p {{
      margin: 0 0 18px 0;
      color: #5b6678;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid rgba(31, 41, 55, 0.08);
      vertical-align: top;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      background: #e6eef8;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    tr:nth-child(even) td {{
      background: rgba(230, 238, 248, 0.28);
    }}
    td:last-child {{
      line-height: 1.55;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html.escape(title)}</h1>
    <p>Ejemplos ordenados por errores de alta confianza para acelerar el debugging del modelo.</p>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Real</th>
          <th>Prediccion</th>
          <th>Confianza</th>
          <th>Texto</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>
</body>
</html>"""


def save_error_analysis_html(errors: list[dict], output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_error_analysis_html(errors), encoding="utf-8")
    return str(path)


def _polyline_points(values: list[float], *, width: int, height: int, padding: int) -> str:
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        hi = lo + 1.0
    usable_w = max(width - padding * 2, 1)
    usable_h = max(height - padding * 2, 1)
    coords: list[str] = []
    for idx, value in enumerate(values):
        x = padding if len(values) == 1 else padding + (idx / (len(values) - 1)) * usable_w
        y = padding + (1.0 - ((value - lo) / (hi - lo))) * usable_h
        coords.append(f"{x:.1f},{y:.1f}")
    return " ".join(coords)


def _render_svg_chart(title: str, series: dict[str, list[float]], colors: dict[str, str]) -> str:
    width = 760
    height = 260
    padding = 26
    all_values = [value for values in series.values() for value in values]
    if not all_values:
        return ""
    lo = min(all_values)
    hi = max(all_values)
    if math.isclose(lo, hi):
        hi = lo + 1.0

    legend = "".join(
        "<div class='legend-item'>"
        f"<span class='legend-dot' style='background:{html.escape(colors[name])}'></span>"
        f"{html.escape(name)}</div>"
        for name in series
    )

    polylines = []
    for name, values in series.items():
        points = _polyline_points(values, width=width, height=height, padding=padding)
        polylines.append(
            f"<polyline fill='none' stroke='{html.escape(colors[name])}' stroke-width='3' points='{points}' />"
        )

    return (
        "<div class='chart-card'>"
        f"<div class='chart-title'>{html.escape(title)}</div>"
        f"<div class='legend'>{legend}</div>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' rx='20' fill='#ffffff' />"
        f"<line x1='{padding}' y1='{padding}' x2='{padding}' y2='{height - padding}' stroke='#d5deea' />"
        f"<line x1='{padding}' y1='{height - padding}' x2='{width - padding}' y2='{height - padding}' stroke='#d5deea' />"
        + "".join(polylines) +
        "</svg>"
        f"<div class='range'>min={lo:.4f} | max={hi:.4f}</div>"
        "</div>"
    )


def render_learning_curves_html(history: list[dict]) -> str:
    train_loss = [float(row.get("train_loss", 0.0)) for row in history]
    val_loss = [float(row.get("val_loss", 0.0)) for row in history]
    train_acc = [float(row.get("train_acc", 0.0)) for row in history]
    val_acc = [float(row.get("val_acc", 0.0)) for row in history]
    train_f1 = [float(row.get("train_f1", 0.0)) for row in history]
    val_f1 = [float(row.get("val_f1", 0.0)) for row in history]

    loss_chart = _render_svg_chart(
        "Loss por epoca",
        {"train_loss": train_loss, "val_loss": val_loss},
        {"train_loss": "#f08700", "val_loss": "#005f73"},
    )
    acc_chart = _render_svg_chart(
        "Accuracy y F1 por epoca",
        {"train_acc": train_acc, "val_acc": val_acc, "train_f1": train_f1, "val_f1": val_f1},
        {"train_acc": "#4f772d", "val_acc": "#1d3557", "train_f1": "#c44536", "val_f1": "#6c5ce7"},
    )

    rows = []
    for row in history:
        rows.append(
            "<tr>"
            f"<td>{int(row.get('epoch', 0))}</td>"
            f"<td>{float(row.get('train_loss', 0.0)):.4f}</td>"
            f"<td>{float(row.get('val_loss', 0.0)):.4f}</td>"
            f"<td>{float(row.get('train_acc', 0.0)):.4f}</td>"
            f"<td>{float(row.get('val_acc', 0.0)):.4f}</td>"
            f"<td>{float(row.get('train_f1', 0.0)):.4f}</td>"
            f"<td>{float(row.get('val_f1', 0.0)):.4f}</td>"
            "</tr>"
        )

    history_json = html.escape(json.dumps(history, ensure_ascii=False, indent=2))
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Curvas de aprendizaje</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: linear-gradient(180deg, #f5f8fc 0%, #fbf8f3 100%);
      color: #17324d;
    }}
    .wrap {{
      max-width: 1120px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 8px 0;
    }}
    .subtitle {{
      margin: 0 0 18px 0;
      color: #5a6f83;
    }}
    .grid {{
      display: grid;
      gap: 18px;
    }}
    .chart-card {{
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid rgba(23, 50, 77, 0.08);
      border-radius: 24px;
      padding: 18px;
      box-shadow: 0 18px 50px rgba(23, 50, 77, 0.08);
    }}
    .chart-title {{
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      font-size: 13px;
      color: #5a6f83;
      margin-bottom: 8px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .legend-dot {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }}
    .range {{
      margin-top: 10px;
      font-size: 12px;
      color: #697c8e;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: rgba(255, 255, 255, 0.95);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 18px 50px rgba(23, 50, 77, 0.08);
    }}
    th, td {{
      padding: 11px 12px;
      border-bottom: 1px solid rgba(23, 50, 77, 0.08);
      text-align: left;
      font-size: 14px;
    }}
    th {{
      background: #e7eef6;
    }}
    pre {{
      margin: 0;
      padding: 16px;
      border-radius: 18px;
      background: #102033;
      color: #d9ebff;
      overflow: auto;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Curvas de aprendizaje</h1>
    <p class="subtitle">Seguimiento de loss, accuracy y F1 para entender convergencia, overfitting y estabilidad.</p>
    <div class="grid">
      {loss_chart}
      {acc_chart}
      <table>
        <thead>
          <tr>
            <th>Epoca</th>
            <th>Train loss</th>
            <th>Val loss</th>
            <th>Train acc</th>
            <th>Val acc</th>
            <th>Train F1</th>
            <th>Val F1</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <pre>{history_json}</pre>
    </div>
  </div>
</body>
</html>"""


def _save_matplotlib_curves(history: list[dict], output_path: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    epochs = [int(row.get("epoch", 0)) for row in history]
    train_loss = [float(row.get("train_loss", 0.0)) for row in history]
    val_loss = [float(row.get("val_loss", 0.0)) for row in history]
    train_acc = [float(row.get("train_acc", 0.0)) for row in history]
    val_acc = [float(row.get("val_acc", 0.0)) for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train_acc")
    axes[1].plot(epochs, val_acc, label="val_acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def save_learning_curves(
    history: list[dict],
    output_path: str | Path,
    *,
    png_path: str | Path | None = None,
) -> list[str]:
    html_path = Path(output_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_learning_curves_html(history), encoding="utf-8")

    created = [str(html_path)]
    if png_path:
        png_file = _save_matplotlib_curves(history, Path(png_path))
        if png_file:
            created.append(png_file)
    return created
