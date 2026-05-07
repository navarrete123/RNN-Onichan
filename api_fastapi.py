"""
api_fastapi.py - API REST opcional con FastAPI.
"""

from __future__ import annotations

from inferencia_avanzada import load_inference_bundle, predict_texts_detailed


def create_app(cfg):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI no esta instalado. Instala 'fastapi' y 'uvicorn' para usar la API."
        ) from exc

    bundle = load_inference_bundle(cfg)
    app = FastAPI(title="RNN Sentiment API", version="1.0.0")

    class PredictRequest(BaseModel):
        text: str

    class PredictResponse(BaseModel):
        sentiment: str
        confidence: float
        probabilities: list[float]
        attention: list[float]
        tokens: list[str]

    @app.get("/health")
    def health():
        return {"status": "ok", "checkpoint": bundle.checkpoint_path}

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest):
        result = predict_texts_detailed(bundle, [payload.text])[0]
        return PredictResponse(
            sentiment=result["label_name"],
            confidence=float(result["confidence"]),
            probabilities=[float(value) for value in result["probabilities"]],
            attention=[float(value) for value in result["attention"]],
            tokens=[str(token) for token in result["tokens"]],
        )

    return app


def run_api(cfg, *, host: str = "127.0.0.1", port: int = 8000):
    app = create_app(cfg)
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn no esta instalado. Instala 'uvicorn' para levantar la API."
        ) from exc
    uvicorn.run(app, host=host, port=port)
