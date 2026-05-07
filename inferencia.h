/**
 * inferencia.h
 * ─────────────────────────────────────────────────────────────────
 * Motor de inferencia RNN para Arduino con TFLite Micro.
 *
 * Responsabilidades:
 *   - Tokenizar texto desde buffer de caracteres (vocab embebido)
 *   - Preparar el tensor de entrada (token IDs + longitud)
 *   - Invocar el interprete TFLite
 *   - Devolver prediccion + confianza + etiquetas activas
 *
 * Compatibilidad:
 *   Arduino Nicla Vision (STM32H747, Cortex-M7 @ 480 MHz, 1MB SRAM)
 *   Arduino Nano 33 BLE Sense (nRF52840, Cortex-M4 @ 64 MHz, 256 KB SRAM)
 *   Arduino Nano 33 BLE Sense Rev2
 */

#pragma once

#include <Arduino.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// ── Configuracion del modelo (debe coincidir con Python) ──────────
#define RNN_SEQ_LEN       64      // longitud de secuencia exportada
#define RNN_NUM_CLASSES    2      // 2=binario, 20=newsgroups, etc.
#define RNN_VOCAB_SIZE  1024      // vocabulario embebido reducido
#define RNN_TENSOR_ARENA_KB 80    // RAM para TFLite (ajustar segun placa)

// ── Resultado de inferencia ───────────────────────────────────────
struct ResultadoRNN {
  uint8_t  clase_predicha;                   // indice de la clase ganadora
  float    probabilidades[RNN_NUM_CLASSES];  // prob por clase (0..1)
  float    confianza;                         // prob de la clase ganadora
  bool     valido;                            // false si hubo error
  char     etiqueta[32];                      // nombre de la clase
};

// ── Tabla de vocabulario embebido ────────────────────────────────
// Vocabulario reducido de las 1024 palabras mas frecuentes.
// Generado automaticamente por pipeline/4_generar_vocab_h.py
// (incluido al final de este archivo como fallback minimo)
struct EntradaVocab {
  const char* token;
  uint16_t    id;
};

// Vocabulario de sentimiento (top 60 palabras para demo)
// En produccion este array viene del archivo vocab_embebido.h
static const EntradaVocab VOCAB_TABLA[] PROGMEM = {
  {"<PAD>",      0}, {"<UNK>",       1},
  {"the",        2}, {"a",           3}, {"and",     4}, {"of",      5},
  {"to",         6}, {"is",          7}, {"in",      8}, {"it",      9},
  {"i",         10}, {"this",       11}, {"that",   12}, {"was",    13},
  {"good",      14}, {"great",      15}, {"bad",    16}, {"awful",  17},
  {"terrible",  18}, {"amazing",    19}, {"love",   20}, {"hate",   21},
  {"boring",    22}, {"excellent",  23}, {"poor",   24}, {"best",   25},
  {"worst",     26}, {"movie",      27}, {"film",   28}, {"story",  29},
  {"acting",    30}, {"actor",      31}, {"plot",   32}, {"scene",  33},
  {"funny",     34}, {"hilarious",  35}, {"scary",  36}, {"dark",   37},
  {"beautiful", 38}, {"touching",   39}, {"waste",  40}, {"time",   41},
  {"watch",     42}, {"recommend",  43}, {"enjoy",  44}, {"like",   45},
  {"dislike",   46}, {"never",      47}, {"always", 48}, {"just",   49},
  {"really",    50}, {"very",       51}, {"not",    52}, {"no",     53},
  {"yes",       54}, {"but",        55}, {"however",56}, {"also",   57},
  {"even",      58}, {"though",     59},
};
static const uint16_t VOCAB_SIZE_EMBEBIDO =
    sizeof(VOCAB_TABLA) / sizeof(VOCAB_TABLA[0]);

// Nombres de las clases
static const char* NOMBRES_CLASES[RNN_NUM_CLASSES] = {
  "negativo",   // clase 0
  "positivo",   // clase 1
};


// ── Clase principal ───────────────────────────────────────────────

class MotorRNN {
public:
  // Constructor
  MotorRNN() : interpreter_(nullptr), arena_(nullptr) {}

  // ── Inicializacion ────────────────────────────────────────────
  bool iniciar(const uint8_t* model_data, size_t model_size) {
    // Verificar el modelo
    model_ = tflite::GetModel(model_data);
    if (model_->version() != TFLITE_SCHEMA_VERSION) {
      Serial.println("[RNN] ERROR: version del modelo incompatible");
      return false;
    }

    // Resolver de operaciones — AllOpsResolver incluye todas las ops
    // Para produccion usar MicroMutableOpResolver con solo las necesarias
    static tflite::AllOpsResolver resolver;

    // Arena de tensor (memoria para activaciones durante inferencia)
    arena_ = new uint8_t[RNN_TENSOR_ARENA_KB * 1024];
    if (!arena_) {
      Serial.println("[RNN] ERROR: no hay RAM suficiente para el arena");
      return false;
    }

    // Crear interprete
    static tflite::MicroInterpreter static_interpreter(
        model_, resolver, arena_, RNN_TENSOR_ARENA_KB * 1024);
    interpreter_ = &static_interpreter;

    // Asignar tensores
    TfLiteStatus status = interpreter_->AllocateTensors();
    if (status != kTfLiteOk) {
      Serial.println("[RNN] ERROR: AllocateTensors fallo");
      return false;
    }

    // Guardar punteros a tensores de entrada/salida
    input_tokens_  = interpreter_->input(0);
    input_lengths_ = interpreter_->input(1);
    output_logits_ = interpreter_->output(0);

    Serial.print("[RNN] Modelo cargado. Arena usada: ");
    Serial.print(interpreter_->arena_used_bytes() / 1024);
    Serial.println(" KB");

    return true;
  }

  // ── Tokenizacion ──────────────────────────────────────────────
  /**
   * Convierte texto a secuencia de token IDs.
   * Lowercases automaticamente, ignora puntuacion.
   *
   * @param texto    Texto de entrada (null-terminated)
   * @param out_ids  Buffer de salida (minimo RNN_SEQ_LEN uint16_t)
   * @return         Numero de tokens encontrados (<= RNN_SEQ_LEN)
   */
  uint16_t tokenizar(const char* texto, uint16_t* out_ids) {
    char     word_buf[64];
    uint16_t word_len  = 0;
    uint16_t n_tokens  = 0;
    const char* ptr    = texto;

    while (*ptr != '\0' && n_tokens < RNN_SEQ_LEN) {
      char c = *ptr++;

      // Lowercasing ASCII
      if (c >= 'A' && c <= 'Z') c += 32;

      bool es_letra = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');

      if (es_letra && word_len < sizeof(word_buf) - 1) {
        word_buf[word_len++] = c;
      } else if (word_len > 0) {
        // Fin de palabra — buscar en vocabulario
        word_buf[word_len] = '\0';
        out_ids[n_tokens++] = buscar_token(word_buf);
        word_len = 0;
      }
    }
    // Ultima palabra
    if (word_len > 0 && n_tokens < RNN_SEQ_LEN) {
      word_buf[word_len] = '\0';
      out_ids[n_tokens++] = buscar_token(word_buf);
    }

    // Rellenar con PAD
    for (uint16_t i = n_tokens; i < RNN_SEQ_LEN; i++) {
      out_ids[i] = 0;  // PAD
    }

    return n_tokens;
  }

  // ── Inferencia ────────────────────────────────────────────────
  /**
   * Clasifica texto directamente.
   * Tokeniza y corre el modelo en un solo paso.
   *
   * @param texto  Texto a clasificar
   * @return       ResultadoRNN con prediccion y probabilidades
   */
  ResultadoRNN clasificar(const char* texto) {
    ResultadoRNN resultado = {};

    if (!interpreter_) {
      resultado.valido = false;
      return resultado;
    }

    // Tokenizar
    uint16_t token_ids[RNN_SEQ_LEN];
    uint16_t n = tokenizar(texto, token_ids);

    return clasificar_tokens(token_ids, n);
  }

  /**
   * Clasifica a partir de token IDs ya preparados.
   * Util si el tokenizador corre en otro modulo.
   */
  ResultadoRNN clasificar_tokens(const uint16_t* token_ids, uint16_t longitud) {
    ResultadoRNN resultado = {};
    resultado.valido = false;

    if (!interpreter_) return resultado;

    // Rellenar tensor de entrada: tokens
    for (uint16_t i = 0; i < RNN_SEQ_LEN; i++) {
      int8_t val = (i < longitud)
                   ? (int8_t)(token_ids[i] % 128)  // escalar a INT8
                   : 0;                              // PAD = 0
      input_tokens_->data.int8[i] = val;
    }

    // Tensor de longitud
    input_lengths_->data.int32[0] = (int32_t)longitud;

    // Ejecutar
    TfLiteStatus status = interpreter_->Invoke();
    if (status != kTfLiteOk) {
      Serial.println("[RNN] ERROR: Invoke() fallo");
      return resultado;
    }

    // Leer salida y dequantizar a float
    float max_prob  = -1.0f;
    uint8_t max_cls = 0;

    float scale      = output_logits_->params.scale;
    int   zero_point = output_logits_->params.zero_point;

    for (int c = 0; c < RNN_NUM_CLASSES; c++) {
      int8_t  raw  = output_logits_->data.int8[c];
      float   prob = (raw - zero_point) * scale;

      // Aplicar softmax simplificado (el modelo puede ya tener softmax)
      resultado.probabilidades[c] = prob;
      if (prob > max_prob) {
        max_prob  = prob;
        max_cls   = c;
      }
    }

    // Normalizar a [0,1] si son logits (sin softmax)
    float suma = 0.0f;
    for (int c = 0; c < RNN_NUM_CLASSES; c++) suma += expf(resultado.probabilidades[c]);
    for (int c = 0; c < RNN_NUM_CLASSES; c++) resultado.probabilidades[c] = expf(resultado.probabilidades[c]) / suma;

    resultado.clase_predicha = max_cls;
    resultado.confianza      = resultado.probabilidades[max_cls];
    strncpy(resultado.etiqueta, NOMBRES_CLASES[max_cls], sizeof(resultado.etiqueta) - 1);
    resultado.valido = true;

    return resultado;
  }

  // ── Informacion de debug ──────────────────────────────────────
  void imprimir_info() {
    if (!interpreter_) { Serial.println("[RNN] No iniciado"); return; }
    Serial.print("[RNN] Arena usada  : "); Serial.print(interpreter_->arena_used_bytes()); Serial.println(" bytes");
    Serial.print("[RNN] Num inputs   : "); Serial.println(interpreter_->inputs_size());
    Serial.print("[RNN] Num outputs  : "); Serial.println(interpreter_->outputs_size());
  }

  ~MotorRNN() {
    delete[] arena_;
  }

private:
  // ── Busqueda de token ─────────────────────────────────────────
  uint16_t buscar_token(const char* word) {
    for (uint16_t i = 0; i < VOCAB_SIZE_EMBEBIDO; i++) {
      // Comparar con string en PROGMEM
      if (strcmp_P(word, (PGM_P)pgm_read_ptr(&VOCAB_TABLA[i].token)) == 0) {
        return (uint16_t)pgm_read_word(&VOCAB_TABLA[i].id);
      }
    }
    return 1;  // <UNK>
  }

  const tflite::Model*       model_;
  tflite::MicroInterpreter*  interpreter_;
  uint8_t*                   arena_;
  TfLiteTensor*              input_tokens_;
  TfLiteTensor*              input_lengths_;
  TfLiteTensor*              output_logits_;
};
