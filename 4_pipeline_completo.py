"""
pipeline/4_pipeline_completo.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ejecuta las 3 etapas en orden:
  1. PyTorch -> ONNX FP32
  2. ONNX FP32 -> ONNX INT8
  3. ONNX INT8 -> TFLite -> Header C para Arduino

Uso rapido:
  python pipeline/4_pipeline_completo.py

Con opciones:
  python pipeline/4_pipeline_completo.py \
      --checkpoint mejor_modelo.pt \
      --seq-len 64 \
      --quant-mode dynamic

Requisitos:
  pip install torch onnx onnxruntime onnx-tf tensorflow
  (opcional) pip install onnxsim
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ETAPAS = [
    ("1_exportar_onnx.py",    "PyTorch → ONNX FP32"),
    ("2_cuantizar_int8.py",   "ONNX FP32 → ONNX INT8"),
    ("3_convertir_tflite.py", "ONNX INT8 → TFLite → Header C"),
]


def run_step(script: str, extra_args: list[str]) -> bool:
    script_path = Path(__file__).parent / script
    cmd = [sys.executable, str(script_path)] + extra_args
    print(f"\n  Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser(description="Pipeline completo PyTorch -> Arduino")
    p.add_argument("--checkpoint",   default="mejor_modelo.pt")
    p.add_argument("--seq-len",      type=int, default=64,
                   help="Longitud de secuencia para MCU (recomendado: 32-64)")
    p.add_argument("--quant-mode",   default="dynamic",
                   choices=["dynamic", "static"])
    p.add_argument("--simplify",     action="store_true")
    p.add_argument("--weights-only", action="store_true")
    p.add_argument("--from-step",    type=int, default=1, choices=[1, 2, 3])
    args = p.parse_args()

    print("=" * 60)
    print("  PIPELINE COMPLETO: PyTorch → Arduino TFLite Micro")
    print("=" * 60)
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  seq_len    : {args.seq_len}")
    print(f"  quant_mode : {args.quant_mode}")

    t_total = time.time()
    resultados = []

    # Argumentos por etapa
    args_etapas = [
        [  # Etapa 1
            "--checkpoint", args.checkpoint,
            "--output",     "artifacts/modelo.onnx",
            "--seq-len",    str(args.seq_len),
        ],
        [  # Etapa 2
            "--input",  "artifacts/modelo.onnx",
            "--output", "artifacts/modelo_int8.onnx",
            "--mode",   args.quant_mode,
            "--checkpoint", args.checkpoint,
        ],
        [  # Etapa 3
            "--input",      "artifacts/modelo_int8.onnx",
            "--savedmodel", "artifacts/savedmodel",
            "--tflite",     "artifacts/modelo.tflite",
            "--header",     "arduino/rnn_sensor/model_data.h",
            "--seq-len",    str(args.seq_len),
            "--checkpoint", args.checkpoint,
        ] + (["--simplify"] if args.simplify else [])
          + (["--weights-only"] if args.weights_only else []),
    ]

    for i, (script, desc) in enumerate(ETAPAS):
        if i + 1 < args.from_step:
            resultados.append((desc, "OMITIDA", 0))
            continue

        print(f"\n{'─'*60}")
        print(f"  ETAPA {i+1}: {desc}")
        print(f"{'─'*60}")

        t0 = time.time()
        ok = run_step(script, args_etapas[i])
        elapsed = time.time() - t0
        resultados.append((desc, "OK" if ok else "ERROR", elapsed))

        if not ok:
            print(f"\n  ERROR en etapa {i+1}. Abortando pipeline.")
            break

    print(f"\n{'='*60}")
    print("  RESUMEN")
    print(f"{'='*60}")
    for desc, status, t in resultados:
        ico = "✓" if status == "OK" else ("→" if status == "OMITIDA" else "✗")
        print(f"  {ico} {desc:<35} {status:>8}  {t:.0f}s")
    print(f"{'─'*60}")
    print(f"  Tiempo total: {time.time() - t_total:.0f}s")

    if all(s in ("OK", "OMITIDA") for _, s, _ in resultados):
        print("\n  Pipeline completado.")
        print("\n  Archivos generados:")
        for path in [
            "artifacts/modelo.onnx",
            "artifacts/modelo_int8.onnx",
            "artifacts/modelo.tflite",
            "arduino/rnn_sensor/model_data.h",
        ]:
            if Path(path).exists():
                size = Path(path).stat().st_size
                print(f"    {path:<45} {size/1024:>7.0f} KB")

        print("\n  Siguiente paso:")
        print("    1. Abre Arduino IDE")
        print("    2. Instala: Arduino_TensorFlowLite (Library Manager)")
        print("    3. Abre arduino/rnn_sensor/rnn_sensor.ino")
        print("    4. Selecciona tu placa (Nicla Vision / Nano 33 BLE Sense)")
        print("    5. Upload!")


if __name__ == "__main__":
    main()
