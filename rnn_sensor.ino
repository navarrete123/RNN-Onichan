/**
 * rnn_sensor.ino
 * ─────────────────────────────────────────────────────────────────
 * Pipeline completo: Sensores → RNN (TFLite Micro) → Actuadores
 *
 * Hardware soportado:
 *   • Arduino Nicla Vision         (STM32H747, Cortex-M7 @ 480 MHz)
 *   • Arduino Nano 33 BLE Sense    (nRF52840, Cortex-M4 @ 64 MHz)
 *   • Arduino Nano 33 BLE Sense Rev2
 *
 * Sensores leidos:
 *   • Temperatura    (HTS221 / HS3003)
 *   • Humedad        (HTS221 / HS3003)
 *   • Microfono PDM  (texto transcripto via Serial o buffer interno)
 *   • Boton          (entrada digital D2)
 *   • Texto via Serial (para pruebas sin sensor)
 *
 * Actuadores:
 *   • LED RGB        (segun clase predicha)
 *   • Servomotor     (angulo proporcional a confianza)
 *   • Serial Monitor (resultado + probabilidades + atencion)
 *
 * Dependencias (instalar en Library Manager):
 *   - Arduino_TensorFlowLite   (TFLite Micro)
 *   - Arduino_HTS221           (temperatura/humedad Nano 33 BLE)
 *   - Servo                    (servomotor)
 *
 * Uso:
 *   1. Completa el pipeline Python: python pipeline/4_pipeline_completo.py
 *   2. Copia model_data.h generado a esta carpeta
 *   3. Sube a la placa
 *   4. Abre Serial Monitor a 115200 baud
 *   5. Escribe texto y presiona Enter, o mantén D2 para leer sensores
 */

#include <Arduino.h>
#include <Servo.h>

// ── Archivos locales ──────────────────────────────────────────────
#include "model_data.h"   // generado por pipeline/3_convertir_tflite.py
#include "inferencia.h"   // motor de inferencia RNN

// ── Deteccion de placa ────────────────────────────────────────────
#if defined(ARDUINO_NICLA_VISION)
  #define PLACA_NOMBRE    "Arduino Nicla Vision"
  #define LED_ROJO        LEDR
  #define LED_VERDE       LEDG
  #define LED_AZUL        LEDB
  #define LED_ACTIVO_BAJO true  // en Nicla el LED se activa con LOW
  // Nicla Vision usa el OV5647 para camara pero aqui leemos serial
  #define TIENE_HTS221    false

#elif defined(ARDUINO_ARDUINO_NANO33BLE)
  #include <Arduino_HTS221.h>
  #define PLACA_NOMBRE    "Arduino Nano 33 BLE Sense"
  #define LED_ROJO        22   // LEDR en Nano 33 BLE
  #define LED_VERDE       23   // LEDG
  #define LED_AZUL        24   // LEDB
  #define LED_ACTIVO_BAJO true
  #define TIENE_HTS221    true

#else
  // Placa generica / desarrollo en Serial
  #define PLACA_NOMBRE    "Placa generica"
  #define LED_ROJO        13
  #define LED_VERDE       12
  #define LED_AZUL        11
  #define LED_ACTIVO_BAJO false
  #define TIENE_HTS221    false
  #warning "Placa no reconocida — usando pines genericos"
#endif

// ── Pines de actuadores ───────────────────────────────────────────
#define PIN_SERVO          9    // PWM — conectar servo aqui
#define PIN_BOTON          2    // boton para trigger manual

// ── Configuracion del sistema ─────────────────────────────────────
#define SERIAL_BAUD        115200
#define SERVO_ANGULO_MIN   10    // angulo para clase "negativo"
#define SERVO_ANGULO_MAX   170   // angulo para clase "positivo"
#define UMBRAL_CONFIANZA   0.65f // confianza minima para actuar
#define BUFFER_TEXTO       256   // buffer de texto de entrada
#define INTERVALO_SENSOR_MS 5000 // lectura automatica cada 5 segundos

// ── Estado global ─────────────────────────────────────────────────
MotorRNN       motor;
Servo          servo;

char           buf_texto[BUFFER_TEXTO];
uint16_t       buf_idx         = 0;
bool           modelo_listo    = false;
unsigned long  ultima_lectura  = 0;
bool           boton_anterior  = HIGH;

// ── Utilidades LED ────────────────────────────────────────────────

void led_apagar() {
  bool v = LED_ACTIVO_BAJO ? HIGH : LOW;
  digitalWrite(LED_ROJO,  v);
  digitalWrite(LED_VERDE, v);
  digitalWrite(LED_AZUL,  v);
}

void led_color(bool r, bool g, bool b) {
  bool inv = LED_ACTIVO_BAJO;
  digitalWrite(LED_ROJO,  inv ? !r : r);
  digitalWrite(LED_VERDE, inv ? !g : g);
  digitalWrite(LED_AZUL,  inv ? !b : b);
}

void parpadeo_inicio() {
  for (int i = 0; i < 3; i++) {
    led_color(false, false, true); delay(150);
    led_apagar();                  delay(150);
  }
}


// ── Lectura de sensores ───────────────────────────────────────────

/**
 * Construye un texto descriptivo a partir de las lecturas de sensores.
 * El modelo fue entrenado con texto en ingles, asi que usamos ingles.
 *
 * Esta funcion es el "puente" entre el mundo fisico y el modelo NLP:
 * convierte numeros de sensores en frases que el vocabulario entiende.
 */
void leer_sensores_a_texto(char* out, uint16_t max_len) {
  float temperatura = 22.0f;
  float humedad     = 50.0f;

#if TIENE_HTS221
  temperatura = HTS.readTemperature();
  humedad     = HTS.readHumidity();
#endif

  // Construir descripcion textual de las condiciones
  // El vocabulario del modelo incluye estas palabras
  const char* desc_temp =
    (temperatura > 35) ? "very hot temperature high uncomfortable" :
    (temperatura > 28) ? "warm temperature good comfortable" :
    (temperatura > 20) ? "nice comfortable temperature pleasant" :
    (temperatura > 10) ? "cool temperature moderate" :
                          "cold temperature low uncomfortable";

  const char* desc_hum =
    (humedad > 80) ? "very humid wet uncomfortable" :
    (humedad > 60) ? "humid moisture moderate" :
    (humedad > 40) ? "comfortable humidity pleasant" :
                      "dry low humidity uncomfortable";

  snprintf(out, max_len,
    "%s %s temperature %.1f humidity %.0f",
    desc_temp, desc_hum, temperatura, humedad);
}


// ── Actuadores ────────────────────────────────────────────────────

/**
 * Controla el servo y el LED segun el resultado de la inferencia.
 *
 * Logica:
 *   - Clase 0 (negativo / condicion mala):
 *       LED rojo + servo en angulo minimo
 *   - Clase 1 (positivo / condicion buena):
 *       LED verde + servo en angulo maximo
 *   - Confianza baja (< UMBRAL_CONFIANZA):
 *       LED azul + servo en centro (sin accion definitiva)
 *
 *   El angulo del servo tambien es proporcional a la confianza:
 *   mas confianza = angulo mas extremo.
 */
void actuar(const ResultadoRNN& r) {
  if (!r.valido) {
    led_color(true, true, false);  // amarillo = error
    return;
  }

  if (r.confianza < UMBRAL_CONFIANZA) {
    // Confianza baja — LED azul, servo en centro
    led_color(false, false, true);
    servo.write(90);
    return;
  }

  // Angulo proporcional a confianza: 0.5 → 90°, 1.0 → angulo extremo
  float t      = (r.confianza - 0.5f) * 2.0f;  // 0..1
  int   angulo = (int)(SERVO_ANGULO_MIN + t * (SERVO_ANGULO_MAX - SERVO_ANGULO_MIN));

  if (r.clase_predicha == 1) {
    // Positivo — verde, servo hacia adelante
    led_color(false, true, false);
    servo.write(angulo);
  } else {
    // Negativo — rojo, servo hacia atras
    led_color(true, false, false);
    servo.write(SERVO_ANGULO_MAX - angulo + SERVO_ANGULO_MIN);
  }
}


// ── Imprimir resultado ────────────────────────────────────────────

void imprimir_resultado(const char* texto, const ResultadoRNN& r) {
  Serial.println("\n┌─────────────────────────────────┐");
  Serial.print  ("│ Texto   : ");
  // Truncar si es muy largo para el monitor
  char tmp[50];
  strncpy(tmp, texto, 47);
  tmp[47] = '\0';
  if (strlen(texto) > 47) strcat(tmp, "...");
  Serial.println(tmp);

  if (!r.valido) {
    Serial.println("│ ERROR: inferencia fallida        │");
    Serial.println("└─────────────────────────────────┘");
    return;
  }

  Serial.print  ("│ Clase   : "); Serial.println(r.etiqueta);
  Serial.print  ("│ Conf    : "); Serial.print(r.confianza * 100, 1); Serial.println("%");

  Serial.print  ("│ Probs   : ");
  for (int c = 0; c < RNN_NUM_CLASSES; c++) {
    Serial.print(c); Serial.print("=");
    Serial.print(r.probabilidades[c] * 100, 1); Serial.print("%  ");
  }
  Serial.println();

  // Barra visual de confianza
  Serial.print  ("│ ");
  int barras = (int)(r.confianza * 20);
  for (int i = 0; i < 20; i++) Serial.print(i < barras ? "█" : "░");
  Serial.println(" │");
  Serial.println("└─────────────────────────────────┘");
}


// ── Procesar texto ────────────────────────────────────────────────

void procesar_texto(const char* texto) {
  if (strlen(texto) < 3) {
    Serial.println("[!] Texto demasiado corto.");
    return;
  }

  led_color(false, false, true);  // azul = procesando

  unsigned long t0 = micros();
  ResultadoRNN  resultado = motor.clasificar(texto);
  unsigned long dt = micros() - t0;

  imprimir_resultado(texto, resultado);
  Serial.print("[t] Inferencia en "); Serial.print(dt / 1000); Serial.println(" ms");

  actuar(resultado);
}


// ── setup() ──────────────────────────────────────────────────────

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial && millis() < 3000);  // esperar Serial (max 3s)

  Serial.println("\n╔═══════════════════════════════════╗");
  Serial.println  ("║  RNN Sensor - TFLite Micro        ║");
  Serial.print    ("║  Placa: "); Serial.print(PLACA_NOMBRE);
  Serial.println  ("       ║");
  Serial.println  ("╚═══════════════════════════════════╝");

  // Pines
  pinMode(LED_ROJO,  OUTPUT);
  pinMode(LED_VERDE, OUTPUT);
  pinMode(LED_AZUL,  OUTPUT);
  pinMode(PIN_BOTON, INPUT_PULLUP);
  led_apagar();

  // Servo
  servo.attach(PIN_SERVO);
  servo.write(90);  // posicion inicial: centro

  // Sensores de temperatura/humedad
#if TIENE_HTS221
  if (!HTS.begin()) {
    Serial.println("[!] HTS221 no encontrado — solo Serial.");
  } else {
    Serial.println("[OK] HTS221 iniciado");
  }
#endif

  // Cargar modelo TFLite
  Serial.print("\nCargando modelo RNN (");
  Serial.print(g_model_data_len);
  Serial.println(" bytes)...");

  modelo_listo = motor.iniciar(g_model_data, g_model_data_len);

  if (modelo_listo) {
    motor.imprimir_info();
    parpadeo_inicio();
    Serial.println("\n[OK] Sistema listo.");
    Serial.println("     > Escribe texto + Enter para clasificar");
    Serial.println("     > Presiona boton D2 para leer sensores");
    Serial.println("     > Texto de ejemplo: 'this movie was great'");
  } else {
    led_color(true, false, false);  // rojo fijo = error
    Serial.println("[ERROR] Modelo no cargado. Verifica model_data.h");
    while (true) delay(1000);
  }
}


// ── loop() ───────────────────────────────────────────────────────

void loop() {
  if (!modelo_listo) return;

  // ── Lectura de Serial (texto manual) ─────────────────────────
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (buf_idx > 0) {
        buf_texto[buf_idx] = '\0';
        Serial.print("\n> "); Serial.println(buf_texto);
        procesar_texto(buf_texto);
        buf_idx = 0;
      }
    } else if (buf_idx < BUFFER_TEXTO - 1) {
      buf_texto[buf_idx++] = c;
    }
  }

  // ── Boton — lectura de sensores ───────────────────────────────
  bool boton_actual = digitalRead(PIN_BOTON);
  bool boton_presionado = (boton_actual == LOW && boton_anterior == HIGH);
  boton_anterior = boton_actual;

  if (boton_presionado) {
    char texto_sensor[BUFFER_TEXTO];
    leer_sensores_a_texto(texto_sensor, sizeof(texto_sensor));
    Serial.print("\n[SENSOR] "); Serial.println(texto_sensor);
    procesar_texto(texto_sensor);
    ultima_lectura = millis();
  }

  // ── Lectura automatica periodica de sensores ──────────────────
  if (millis() - ultima_lectura > INTERVALO_SENSOR_MS && TIENE_HTS221) {
    char texto_sensor[BUFFER_TEXTO];
    leer_sensores_a_texto(texto_sensor, sizeof(texto_sensor));
    procesar_texto(texto_sensor);
    ultima_lectura = millis();
  }
}
