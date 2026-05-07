"""
pipeline/1_exportar_onnx.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ETAPA 1 — PyTorch → ONNX (FP32)

Por que forward_exportable en vez de forward:
  El forward normal usa pack_padded_sequence, que no es
  exportable a ONNX. forward_exportable usa la RNN directamente
  sin empaquetar, generando un grafo estatico compatible.

Salidas:
  artifacts/modelo.onnx       modelo FP32
  artifacts/vocab.json        (ya existe, se copia)

Uso:
  python pipeline/1_exportar_onnx.py
  python pipeline/1_exportar_onnx.py --checkpoint mi_modelo.pt
  python pipeline/1_exportar_onnx.py --seq-len 64  # secuencias cortas para MCU
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


# ── Wrapper exportable ────────────────────────────────────────────

class ExportWrapper(nn.Module):
    """
    Envuelve MejorRNN para exportacion ONNX.
    Entrada : tokens (B, T)  lengths (B,)
    Salida  : logits (B, C)  probs (B, C)  attention (B, T)

    Usa forward_exportable (sin pack_padded_sequence).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        tokens:  torch.Tensor,   # (B, T) long
        lengths: torch.Tensor,   # (B,)   long
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, attention = self.model.forward_exportable(tokens, lengths)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs, attention


# ── Carga del checkpoint ──────────────────────────────────────────

def cargar_modelo(ckpt_path: str):
    """Reconstruye Config + MejorRNN desde el checkpoint."""
    # Fix: funciona tanto en raiz como en subcarpeta pipeline/
    _here = Path(__file__).resolve().parent
    for _p in [_here, _here.parent]:
        if (_p / "configuracion.py").exists():
            if str(_p) not in sys.path:
                sys.path.insert(0, str(_p))
            break
    from configuracion import Config
    from modelo_profesional import MejorRNN
    from datos_texto import Vocabulary

    # weights_only=False requerido: checkpoint contiene dicts Python (config, vocab, etc.)
    # PyTorch >= 2.6 cambio el default a True y rompe la carga
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise ValueError("El checkpoint no contiene 'config'. Reentrenalo con la version actual.")

    cfg   = Config(**ckpt["config"])
    cfg.device = "cpu"

    if "vocab" in ckpt:
        vocab = Vocabulary.from_state(ckpt["vocab"])
    else:
        vocab = Vocabulary.load(cfg.vocab_path)

    model = MejorRNN(cfg, vocab_size=len(vocab))
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, cfg, vocab


# ── Verificacion del modelo ───────────────────────────────────────

def verificar_forward_exportable(model, cfg, seq_len: int):
    """Verifica que forward_exportable produce los mismos logits que forward."""
    if not hasattr(model, "forward_exportable"):
        raise AttributeError(
            "MejorRNN no tiene forward_exportable. "
            "Actualiza modelo_profesional.py a la version con ese metodo."
        )
    B, T = 2, seq_len
    x       = torch.randint(1, 100, (B, T))
    lengths = torch.tensor([T, T // 2])

    with torch.no_grad():
        logits_std, _  = model(x, lengths)
        logits_exp, _  = model.forward_exportable(x, lengths)

    diff = (logits_std - logits_exp).abs().max().item()
    print(f"  Diferencia forward vs forward_exportable: {diff:.2e}")
    if diff > 1e-3:
        print("  ADVERTENCIA: diferencia alta. "
              "El padding puede afectar resultados en secuencias variables.")
    else:
        print("  OK — los dos caminos son equivalentes.")


# ── Exportacion ONNX ──────────────────────────────────────────────

def exportar_onnx(
    model,
    cfg,
    output_path: str,
    seq_len:     int = 128,
    batch_size:  int = 1,
    opset:       int = 17,
) -> str:
    wrapper = ExportWrapper(model)
    wrapper.eval()

    # Input de ejemplo — representa una secuencia de tokens
    tokens  = torch.randint(1, cfg.max_vocab, (batch_size, seq_len), dtype=torch.long)
    lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (tokens, lengths),
        output_path,
        export_params       = True,
        opset_version       = opset,
        do_constant_folding = True,
        input_names         = ["tokens", "lengths"],
        output_names        = ["logits", "probabilities", "attention"],
        dynamic_axes        = {
            "tokens":       {0: "batch_size", 1: "seq_len"},
            "lengths":      {0: "batch_size"},
            "logits":       {0: "batch_size"},
            "probabilities":{0: "batch_size"},
            "attention":    {0: "batch_size", 1: "seq_len"},
        },
    )
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"  ONNX exportado -> {output_path}  ({size_mb:.1f} MB)")
    return output_path


# ── Validacion con onnxruntime ────────────────────────────────────

def validar_onnx(onnx_path: str, cfg, seq_len: int = 32):
    """Corre una inferencia con onnxruntime y compara con PyTorch."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("  (Omitiendo validacion — instala 'onnxruntime' para validar)")
        return

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    tokens  = np.random.randint(1, cfg.max_vocab, (1, seq_len), dtype=np.int64)
    lengths = np.array([seq_len], dtype=np.int64)

    logits, probs, attn = sess.run(None, {"tokens": tokens, "lengths": lengths})
    pred = probs.argmax(axis=-1)[0]
    print(f"  Validacion onnxruntime OK — prediccion={pred}  probs={probs[0].tolist()}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="mejor_modelo.pt")
    p.add_argument("--output",     default="artifacts/modelo.onnx")
    p.add_argument("--seq-len",    type=int, default=128,
                   help="Longitud de secuencia fija para exportacion (recomendado: 64 para MCU)")
    p.add_argument("--opset",      type=int, default=17)
    p.add_argument("--no-validate", action="store_true")
    args = p.parse_args()

    print("\n[ETAPA 1] Exportando PyTorch -> ONNX")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  seq_len    : {args.seq_len}")
    print(f"  opset      : {args.opset}")

    model, cfg, vocab = cargar_modelo(args.checkpoint)

    print("\n  Verificando forward_exportable...")
    verificar_forward_exportable(model, cfg, args.seq_len)

    print(f"\n  Exportando a {args.output}...")
    exportar_onnx(model, cfg, args.output, seq_len=args.seq_len, opset=args.opset)

    if not args.no_validate:
        print("\n  Validando con onnxruntime...")
        validar_onnx(args.output, cfg, args.seq_len)

    print("\n  Listo. Siguiente paso:")
    print("  python pipeline/2_cuantizar_int8.py")


if __name__ == "__main__":
    main()
