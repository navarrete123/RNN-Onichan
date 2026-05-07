"""
pipeline/2_cuantizar_int8.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ETAPA 2 — ONNX FP32 → ONNX INT8 (cuantizacion estatica)

Por que INT8:
  FP32  → ~37 MB, requiere FPU completa
  INT8  → ~10 MB, corre en ARM Cortex-M4/M7 sin FPU dedicada
  La perdida de precision es tipicamente < 1% en accuracy

Tipos de cuantizacion disponibles:
  - dinamica   : rapida, sin datos de calibracion, buena para RNN
  - estatica   : requiere datos de calibracion, mas precisa para CNN/MLP
  - QAT        : requiere reentrenamiento, mejor precision posible

Para RNN se recomienda DINAMICA porque los estados ocultos
tienen rango variable segun la entrada.

Uso:
  pip install onnxruntime onnx
  python pipeline/2_cuantizar_int8.py
  python pipeline/2_cuantizar_int8.py --mode static --calib-data calibracion.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cuantizar_dinamico(input_path: str, output_path: str) -> str:
    """
    Cuantizacion dinamica: los pesos se cuantizan a INT8 en disco,
    las activaciones se cuantizan en tiempo de ejecucion.
    Ideal para LSTM/GRU — no necesita datos de calibracion.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input   = input_path,
        model_output  = output_path,
        weight_type   = QuantType.QInt8,
        # Cuantizar operaciones de matriz — clave para LSTM
        extra_options = {
            "MatMulConstBOnly": True,   # solo cuantiza pesos constantes
        },
    )
    return output_path


def cuantizar_estatico(input_path: str, output_path: str, calib_data_path: str) -> str:
    """
    Cuantizacion estatica: requiere datos de calibracion para determinar
    los rangos de cuantizacion de las activaciones.
    Mas precisa pero requiere un dataset representativo.
    """
    import numpy as np
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantType,
        QuantFormat,
        quantize_static,
    )

    class RNNCalibReader(CalibrationDataReader):
        def __init__(self, data_path: str):
            self.data  = np.load(data_path, allow_pickle=True).item()
            self.index = 0
            self.items = list(self.data.items()) if isinstance(self.data, dict) else []

        def get_next(self):
            if not self.items or self.index >= len(self.items[0][1]):
                return None
            batch = {k: v[self.index : self.index + 1] for k, v in self.items}
            self.index += 1
            return batch

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        model_input         = input_path,
        model_output        = output_path,
        calibration_data_reader = RNNCalibReader(calib_data_path),
        weight_type         = QuantType.QInt8,
        activation_type     = QuantType.QInt8,
        quant_format        = QuantFormat.QDQ,
    )
    return output_path


def generar_datos_calibracion(
    checkpoint_path: str,
    n_samples: int = 200,
    seq_len:   int = 128,
    output:    str = "artifacts/calibracion.npy",
) -> str:
    """
    Genera datos de calibracion desde el vocabulario del checkpoint.
    Simula entradas reales con tokens del vocabulario entrenado.
    """
    import numpy as np
    import torch

    # Fix: funciona tanto en raiz como en subcarpeta pipeline/
    _here = Path(__file__).resolve().parent
    for _p in [_here, _here.parent]:
        if (_p / "configuracion.py").exists():
            if str(_p) not in sys.path:
                sys.path.insert(0, str(_p))
            break
    from datos_texto import Vocabulary

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocab = (Vocabulary.from_state(ckpt["vocab"]) if "vocab" in ckpt
             else Vocabulary.load("vocab.json"))

    vocab_size = len(vocab)
    rng = np.random.default_rng(42)

    tokens  = rng.integers(1, vocab_size, (n_samples, seq_len), dtype=np.int64)
    lengths = rng.integers(seq_len // 4, seq_len + 1, n_samples, dtype=np.int64)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.save(output, {"tokens": tokens, "lengths": lengths})
    print(f"  Datos de calibracion generados -> {output} ({n_samples} muestras)")
    return output


def comparar_modelos(fp32_path: str, int8_path: str, n_runs: int = 100):
    """Compara accuracy y velocidad entre FP32 e INT8."""
    import numpy as np
    import time

    try:
        import onnxruntime as ort
    except ImportError:
        print("  (Omitiendo comparacion — instala 'onnxruntime')")
        return

    sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    rng     = np.random.default_rng(42)
    tokens  = rng.integers(1, 1000, (1, 64), dtype=np.int64)
    lengths = np.array([64], dtype=np.int64)
    feed    = {"tokens": tokens, "lengths": lengths}

    # Velocidad
    t0 = time.perf_counter()
    for _ in range(n_runs):
        out_fp32 = sess_fp32.run(None, feed)
    t_fp32 = (time.perf_counter() - t0) / n_runs * 1000

    t0 = time.perf_counter()
    for _ in range(n_runs):
        out_int8 = sess_int8.run(None, feed)
    t_int8 = (time.perf_counter() - t0) / n_runs * 1000

    # Diferencia en probabilidades
    prob_diff = abs(out_fp32[1] - out_int8[1]).max()
    pred_fp32 = out_fp32[1].argmax()
    pred_int8 = out_int8[1].argmax()

    size_fp32 = Path(fp32_path).stat().st_size / 1e6
    size_int8 = Path(int8_path).stat().st_size / 1e6

    print(f"\n  Comparacion FP32 vs INT8")
    print(f"  {'':20s} {'FP32':>10}  {'INT8':>10}  {'Mejora':>10}")
    print(f"  {'Tamano (MB)':20s} {size_fp32:>10.1f}  {size_int8:>10.1f}  "
          f"{size_fp32/size_int8:>9.1f}x")
    print(f"  {'Latencia (ms)':20s} {t_fp32:>10.2f}  {t_int8:>10.2f}  "
          f"{t_fp32/t_int8:>9.1f}x")
    print(f"  {'Prediccion':20s} {pred_fp32:>10}  {pred_int8:>10}")
    print(f"  {'Max prob diff':20s} {'':>10}  {prob_diff:>10.4f}")
    if pred_fp32 == pred_int8:
        print(f"  Prediccion identica en esta muestra OK")
    else:
        print(f"  DIFERENCIA en prediccion — revisar calibracion")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       default="artifacts/modelo.onnx")
    p.add_argument("--output",      default="artifacts/modelo_int8.onnx")
    p.add_argument("--mode",        default="dynamic",
                   choices=["dynamic", "static"])
    p.add_argument("--calib-data",  default=None,
                   help="Ruta a .npy de calibracion (solo --mode static)")
    p.add_argument("--checkpoint",  default="mejor_modelo.pt",
                   help="Usado para generar datos de calibracion si no existen")
    p.add_argument("--seq-len",     type=int, default=128)
    p.add_argument("--no-compare",  action="store_true")
    args = p.parse_args()

    try:
        import onnxruntime  # noqa
    except ImportError:
        print("Instala: pip install onnxruntime onnx")
        sys.exit(1)

    print(f"\n[ETAPA 2] Cuantizando ONNX -> INT8 (modo: {args.mode})")
    print(f"  entrada : {args.input}")
    print(f"  salida  : {args.output}")

    if not Path(args.input).exists():
        print(f"  ERROR: No existe {args.input}. Ejecuta primero la etapa 1.")
        sys.exit(1)

    if args.mode == "dynamic":
        print("\n  Aplicando cuantizacion DINAMICA (recomendada para LSTM)...")
        cuantizar_dinamico(args.input, args.output)

    else:  # static
        calib = args.calib_data or "artifacts/calibracion.npy"
        if not Path(calib).exists():
            print(f"\n  Generando datos de calibracion ({calib})...")
            generar_datos_calibracion(args.checkpoint, seq_len=args.seq_len, output=calib)

        print(f"\n  Aplicando cuantizacion ESTATICA con {calib}...")
        cuantizar_estatico(args.input, args.output, calib)

    size_in  = Path(args.input ).stat().st_size / 1e6
    size_out = Path(args.output).stat().st_size / 1e6
    print(f"\n  FP32 : {size_in:.1f} MB")
    print(f"  INT8 : {size_out:.1f} MB  (reduccion {size_in/size_out:.1f}x)")

    if not args.no_compare:
        comparar_modelos(args.input, args.output)

    print("\n  Listo. Siguiente paso:")
    print("  python pipeline/3_convertir_tflite.py")


if __name__ == "__main__":
    main()
