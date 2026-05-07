/**
 * model_data.h — Datos del modelo TFLite embebido
 * ─────────────────────────────────────────────────────────────────
 * ESTE ARCHIVO ES GENERADO AUTOMATICAMENTE.
 * No editar manualmente.
 *
 * Para regenerar:
 *   python pipeline/4_pipeline_completo.py --seq-len 64
 *
 * El archivo resultante (artifacts/modelo.tflite) se convierte
 * al array C a continuacion mediante:
 *   python pipeline/3_convertir_tflite.py --skip-savedmodel
 *
 * Tamaño esperado segun configuracion del modelo:
 *   FP32 original   ~37 MB  (no apto para MCU)
 *   INT8 cuantizado ~ 9 MB  (Nicla Vision: OK / Nano 33: limite)
 *   INT8 + pruning  ~ 3 MB  (recomendado para Nano 33 BLE)
 *
 * Si el modelo es demasiado grande para tu placa:
 *   1. Reduce hidden_dim a 128 en configuracion.py
 *   2. Reduce num_layers a 1
 *   3. Reduce seq_len a 32 en el pipeline
 *   4. Reentrenar y re-exportar
 */

#pragma once
#include <stdint.h>

// ── PLACEHOLDER ──────────────────────────────────────────────────
// Reemplazar este bloque con el contenido generado por el pipeline.
//
// Ejemplo de como queda despues de la generacion:
//
//   const unsigned int g_model_data_len = 9437184;
//   alignas(8) const uint8_t g_model_data[] = {
//     0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33,
//     ... (miles de lineas) ...
//   };

// Modelo dummy de 8 bytes para compilar sin el modelo real
// (el sistema se detendra en setup() con un mensaje de error)
const unsigned int g_model_data_len = 8;
alignas(8) const uint8_t g_model_data[] = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

// ─────────────────────────────────────────────────────────────────
// INSTRUCCIONES PARA REEMPLAZAR:
//
// 1. Ejecutar:
//      python pipeline/4_pipeline_completo.py
//
// 2. El archivo generado estara en:
//      arduino/rnn_sensor/model_data.h
//
// 3. Copiar ese archivo aqui y subir a la placa.
// ─────────────────────────────────────────────────────────────────
