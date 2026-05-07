"""
pipeline/3_convertir_tflite.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ETAPA 3 — ONNX INT8 → TensorFlow SavedModel → TFLite

Ruta de conversion:
  ONNX → TF SavedModel (via onnx-tf)
  TF SavedModel → TFLite INT8 (via TFLiteConverter)
  TFLite → C header array (para embeber en firmware Arduino)

Uso:
  pip install onnx-tf tensorflow
  python pipeline/3_convertir_tflite.py

  # Si onnx-tf da problemas con ops complejas del LSTM:
  python pipeline/3_convertir_tflite.py --simplify-first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── ONNX → TF SavedModel ─────────────────────────────────────────

def simplificar_onnx(input_path: str, output_path: str) -> str:
    """
    Simplifica el grafo ONNX antes de convertir.
    Fusiona ops, elimina nodos redundantes.
    Requiere: pip install onnxsim
    """
    try:
        import onnx
        from onnxsim import simplify

        model = onnx.load(input_path)
        simplified, ok = simplify(model)
        if not ok:
            print("  Simplificacion parcial — usando modelo original")
            return input_path
        onnx.save(simplified, output_path)
        print(f"  ONNX simplificado -> {output_path}")
        return output_path
    except ImportError:
        print("  (onnxsim no instalado — omitiendo simplificacion)")
        return input_path


def onnx_a_savedmodel(onnx_path: str, savedmodel_path: str) -> str:
    """Convierte ONNX a TensorFlow SavedModel via onnx-tf."""
    try:
        import onnx
        import onnx_tf
        from onnx_tf.backend import prepare
    except ImportError:
        print("Instala: pip install onnx-tf tensorflow")
        sys.exit(1)

    print(f"  Cargando ONNX desde {onnx_path}...")
    model = onnx.load(onnx_path)

    print("  Convirtiendo a TF SavedModel...")
    tf_rep = prepare(model)
    tf_rep.export_graph(savedmodel_path)

    print(f"  SavedModel exportado -> {savedmodel_path}")
    return savedmodel_path


# ── TF SavedModel → TFLite ────────────────────────────────────────

def savedmodel_a_tflite(
    savedmodel_path: str,
    tflite_path:     str,
    *,
    quantize_full_int8: bool = True,
    representative_dataset_fn = None,
) -> str:
    """
    Convierte TF SavedModel a TFLite con cuantizacion INT8 completa.
    Si representative_dataset_fn es None, usa cuantizacion de pesos solamente.
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("Instala: pip install tensorflow")
        sys.exit(1)

    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)

    if quantize_full_int8:
        # Cuantizacion completa INT8 — necesaria para Cortex-M sin FPU
        converter.optimizations          = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        ]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8

        if representative_dataset_fn:
            converter.representative_dataset = representative_dataset_fn
        else:
            print("  ADVERTENCIA: sin dataset representativo, "
                  "la cuantizacion de activaciones es menos precisa.")
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.SELECT_TF_OPS,  # fallback para ops no soportadas
            ]
    else:
        # Solo cuantizacion de pesos (mas compatible)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    Path(tflite_path).parent.mkdir(parents=True, exist_ok=True)
    Path(tflite_path).write_bytes(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"  TFLite generado -> {tflite_path}  ({size_kb:.0f} KB)")
    return tflite_path


# ── Dataset representativo para TFLite ───────────────────────────

def crear_representative_dataset(vocab_size: int, seq_len: int = 64, n: int = 100):
    """
    Genera un dataset representativo de tensores INT8 para la conversion.
    TFLite necesita este dataset para calcular los rangos de cuantizacion.
    """
    import numpy as np

    def generator():
        rng = np.random.default_rng(42)
        for _ in range(n):
            tokens  = rng.integers(0, vocab_size, (1, seq_len), dtype=np.int32)
            lengths = np.array([[seq_len]], dtype=np.int32)
            yield [tokens, lengths]

    return generator


# ── TFLite → Header C ─────────────────────────────────────────────

def tflite_a_header_c(
    tflite_path:  str,
    header_path:  str,
    array_name:   str = "g_model_data",
) -> str:
    """
    Convierte el .tflite a un array C/C++ para embeber en firmware Arduino.

    El header resultante se incluye directamente en el .ino:
        #include "model_data.h"
        // el array g_model_data[] contiene el modelo completo
    """
    data = Path(tflite_path).read_bytes()
    n    = len(data)

    lines = [
        "// AUTO-GENERADO por pipeline/3_convertir_tflite.py",
        "// NO EDITAR MANUALMENTE",
        "//",
        f"// Modelo : {Path(tflite_path).name}",
        f"// Tamano : {n:,} bytes ({n/1024:.1f} KB)",
        "",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"// Almacenado en flash (PROGMEM) para ahorrar RAM en Arduino",
        f"const unsigned int {array_name}_len = {n};",
        "",
        "// Para Arduino Nano 33 BLE / Nicla Vision (ARM Cortex-M4/M7):",
        "// La siguiente directiva coloca el modelo en flash, no en RAM.",
        f"alignas(8) const uint8_t {array_name}[] = {{",
    ]

    # 12 bytes por linea para legibilidad
    row_size = 12
    for i in range(0, n, row_size):
        chunk  = data[i : i + row_size]
        hexstr = ", ".join(f"0x{b:02x}" for b in chunk)
        comma  = "," if i + row_size < n else ""
        lines.append(f"  {hexstr}{comma}")

    lines += ["};", ""]

    Path(header_path).parent.mkdir(parents=True, exist_ok=True)
    Path(header_path).write_text("\n".join(lines), encoding="utf-8")

    print(f"  Header C generado -> {header_path}  ({n:,} bytes / {n/1024:.1f} KB)")
    return header_path


# ── Validar TFLite ────────────────────────────────────────────────

def validar_tflite(tflite_path: str, vocab_size: int, seq_len: int = 64):
    """Corre una inferencia con TFLite Interpreter para verificar el modelo."""
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("  (Omitiendo validacion TFLite — instala tensorflow)")
        return

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Inputs  : {[(d['name'], d['shape'], d['dtype'].__name__) for d in input_details]}")
    print(f"  Outputs : {[(d['name'], d['shape'], d['dtype'].__name__) for d in output_details]}")

    # Input de prueba
    rng = np.random.default_rng(42)
    for detail in input_details:
        name  = detail["name"]
        shape = detail["shape"]
        dtype = detail["dtype"]
        if "token" in name.lower() or shape[-1] == seq_len:
            data = rng.integers(1, vocab_size, shape, dtype=np.int32).astype(dtype)
        else:
            data = np.array([seq_len], dtype=dtype).reshape(shape)
        interpreter.set_tensor(detail["index"], data)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    print(f"  Salida de prueba: {output}")
    print(f"  Validacion TFLite OK")


# ── Main ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",        default="artifacts/modelo_int8.onnx")
    p.add_argument("--savedmodel",   default="artifacts/savedmodel")
    p.add_argument("--tflite",       default="artifacts/modelo.tflite")
    p.add_argument("--header",       default="arduino/rnn_sensor/model_data.h")
    p.add_argument("--seq-len",      type=int, default=64,
                   help="Longitud de secuencia — debe coincidir con lo exportado en etapa 1")
    p.add_argument("--simplify",     action="store_true",
                   help="Simplificar ONNX antes de convertir (requiere onnxsim)")
    p.add_argument("--weights-only", action="store_true",
                   help="Solo cuantizar pesos (mas compatible, menos optimo)")
    p.add_argument("--checkpoint",   default="mejor_modelo.pt",
                   help="Para obtener vocab_size del dataset representativo")
    p.add_argument("--skip-header",  action="store_true")
    args = p.parse_args()

    print("\n[ETAPA 3] Convirtiendo ONNX INT8 -> TFLite -> Header C")

    if not Path(args.input).exists():
        print(f"ERROR: No existe {args.input}. Ejecuta primero la etapa 2.")
        sys.exit(1)

    # Simplificar ONNX opcionalmente
    onnx_to_convert = args.input
    if args.simplify:
        print("\n  Simplificando grafo ONNX...")
        simplified = args.input.replace(".onnx", "_simplified.onnx")
        onnx_to_convert = simplificar_onnx(args.input, simplified)

    # ONNX → TF SavedModel
    print(f"\n  ONNX -> SavedModel ({args.savedmodel})...")
    onnx_a_savedmodel(onnx_to_convert, args.savedmodel)

    # Obtener vocab_size para el dataset representativo
    vocab_size = 30_000  # default
    try:
        import torch
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "vocab" in ckpt and "word2idx" in ckpt["vocab"]:
            vocab_size = len(ckpt["vocab"]["word2idx"])
            print(f"  vocab_size detectado: {vocab_size:,}")
    except Exception:
        print(f"  Usando vocab_size default: {vocab_size:,}")

    # SavedModel → TFLite
    print(f"\n  SavedModel -> TFLite ({args.tflite})...")
    rep_dataset = (None if args.weights_only
                   else crear_representative_dataset(vocab_size, args.seq_len))
    savedmodel_a_tflite(
        args.savedmodel, args.tflite,
        quantize_full_int8      = not args.weights_only,
        representative_dataset_fn = rep_dataset,
    )

    # TFLite → Header C
    if not args.skip_header:
        print(f"\n  TFLite -> Header C ({args.header})...")
        tflite_a_header_c(args.tflite, args.header)

    # Validar
    print("\n  Validando TFLite interpreter...")
    validar_tflite(args.tflite, vocab_size, args.seq_len)

    print(f"\n  Resumen de tamanos:")
    for label, path in [
        ("ONNX FP32 ", "artifacts/modelo.onnx"),
        ("ONNX INT8 ", "artifacts/modelo_int8.onnx"),
        ("TFLite    ", args.tflite),
    ]:
        if Path(path).exists():
            print(f"    {label} : {Path(path).stat().st_size / 1e6:.2f} MB")

    print("\n  Listo. Siguiente paso:")
    print(f"  Copiar arduino/rnn_sensor/ a tu IDE de Arduino y flashear.")


if __name__ == "__main__":
    main()
