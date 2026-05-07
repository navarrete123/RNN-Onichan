# Informe de Resultados - RNN Onichan
## 1. Configuración del Experimento
- **Arquitectura:** LSTM Bidireccional
- **Capas:** 2 | **Hidden Dim:** 256
- **Regularización:** Dropout 0.35, Weight Decay 0.01

## 2. Métricas de Entrenamiento
- **Épocas ejecutadas:** 6
- **Mejor Precisión (Val):** 0.8840
- **Brecha de Generalización:** 0.1181 (⚠️ Detectado)

## 3. Desempeño en Test (Final)
- **Precisión Total:** 0.8504
- **F1-Score Promedio:** 0.8503
- **Pérdida (Loss):** 0.3837

## 4. Conclusiones
- El modelo muestra signos de sobreajuste. Se sugiere aumentar el `attention_dropout` o reducir `hidden_dim`.
- Rendimiento excelente. La combinación de pooling de atención y global es efectiva para este dataset.
