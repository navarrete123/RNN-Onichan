# RNN Sentiment Toolkit

Proyecto de clasificacion de texto con RNN bidireccional, atencion, exportacion y serving.

## Uso rapido

Entrenar con IMDB:

```bash
python main.py
```

Entrenar con datos propios:

```bash
python main.py --train-data datos.csv --text-column text --label-column label
```

Inferencia con visualizador de atencion:

```bash
python main.py --infer-text "This movie was amazing" --attention-html
```

Prediccion batch sobre CSV:

```bash
python main.py --batch-input reviews.csv --batch-output predicciones.csv
```

Exportar a ONNX:

```bash
python main.py --export-onnx
```

Lanzar UI web con Gradio:

```bash
python main.py --launch-gradio
```

Levantar API REST:

```bash
python main.py --serve-api
```

## Features agregadas

- UI con Gradio para clasificar en tiempo real y ver atencion por token.
- API REST con FastAPI en `/predict`.
- Exportacion a ONNX.
- Prediccion batch CSV/TSV.
- Embeddings preentrenados desde archivo local GloVe/FastText.
- Data augmentation con sinonimos, swap y delete aleatorio.
- Entrenamiento de ensembles por seeds.
- HTML interactivo para atencion.
- Analisis de errores de alta confianza.
- Curvas de aprendizaje guardadas en `artifacts/`.
- Tracking opcional con MLflow o WandB.

## Dependencias opcionales

Las integraciones de producto y tracking usan paquetes extra:

```bash
pip install -r requirements-extra.txt
```
