"""
EXPERIMENTO E3 - CNN DESDE CERO, versión mejorada (v2), entrenada en la GPU.

Pregunta que responde: ¿hasta dónde llega una CNN SIN modelo preentrenado con
nuestros 2 125 datos de entrenamiento?

Qué cambia respecto a la CNN del experimento E2 (58.8 % en test) y por qué:

  1. BatchNormalization después de cada Conv2D
     Normaliza las activaciones de cada lote -> el entrenamiento es más estable y
     rápido, y se puede usar un learning rate más alto sin que diverja.

  2. Dos convoluciones por bloque (Conv-BN-ReLU, Conv-BN-ReLU, MaxPool), al estilo VGG
     Dos filtros 3x3 seguidos "ven" una zona de 5x5 con menos parámetros que un 5x5
     y con una ReLU extra en medio -> patrones más complejos.

  3. GlobalAveragePooling2D en lugar de Flatten
     En el experimento E2, Flatten (10x10x128 = 12 800 valores) + Dense(128) tenía ~1.6 M de
     los 1.88 M de parámetros: la red memorizaba posiciones exactas.
     GAP promedia cada mapa de características -> 256 valores, casi sin parámetros,
     y no le importa en qué parte de la foto está el animal.

  4. Entrenar mucho más tiempo (hasta 150 épocas, paciencia 20)
     En el experimento E2 el EarlyStopping cortó en la época 15 con la accuracy de
     entrenamiento en 57 % y todavía subiendo: la red no había terminado de aprender.

  5. GPU (RTX 5070) con Keras 3 + backend PyTorch
     TensorFlow ya no usa GPU en Windows nativo; PyTorch sí. Keras 3 permite cambiar
     el "motor" con la variable KERAS_BACKEND sin tocar el código del modelo.
     El archivo .keras resultante se puede cargar luego con TensorFlow (clasificador.py).

  6. Data augmentation en el pipeline tf.data (CPU, en paralelo) y no dentro del modelo
     Medido: con PyTorch, rotar/zoom/desplazar dentro del modelo costaba ~390 ms por lote
     y la GPU estaba al ~20 %. Moverlo a tf.data bajó cada época de 34 s a 6 s.

Salidas:
  modelos/cnn_propia_v2.keras  + .json
  resultados/curvas_cnn_propia_v2.png
  resultados/historial_cnn_propia_v2.json

Uso:  .venv\\Scripts\\python.exe historial\\src\\exp3_entrenar_cnn_v2.py
"""
import os

# Debe definirse ANTES de importar keras
os.environ.setdefault("KERAS_BACKEND", "torch")

import time

import keras
from keras import layers

from comun import (CLASES, DIR_MODELOS, IMG_SIZE, SEED, capa_aumento_datos,
                   cargar_datasets, guardar_historial, guardar_metadatos)

NOMBRE = "cnn_propia_v2"
# Con $env:DATASET = "dataset_cnn_ampliado" (paso 1 del pipeline final) el modelo se guarda como cnn_propia_v2_ampliado
if os.environ.get("DATASET", "dataset_cnn") != "dataset_cnn":
    NOMBRE += "_" + os.environ["DATASET"].removeprefix("dataset_cnn_").replace("/", "_")
# Semilla de ENTRENAMIENTO (pesos iniciales, orden de lotes, aumento de datos). La división
# train/validación siempre usa SEED=42, así todas las corridas ven las mismas fotos.
# Con $env:SEMILLA = "1" se repite el experimento con otro azar -> cnn_propia_v2_s1
SEMILLA_ENTRENAMIENTO = int(os.environ.get("SEMILLA", SEED))
if SEMILLA_ENTRENAMIENTO != SEED:
    NOMBRE += f"_s{SEMILLA_ENTRENAMIENTO}"
EPOCAS = 150              # máximo; EarlyStopping para antes
LEARNING_RATE = 1e-3
FILTROS = [32, 64, 128, 256]

keras.utils.set_random_seed(SEMILLA_ENTRENAMIENTO)

dispositivo = "CPU"
if keras.backend.backend() == "torch":
    import torch
    if torch.cuda.is_available():
        dispositivo = f"GPU ({torch.cuda.get_device_name(0)})"
print(f"Backend de Keras: {keras.backend.backend()}  |  Dispositivo: {dispositivo}")

# 1. Datos (los mismos que en el experimento E2, salvo que se elija otro con DATASET) -----
train_ds, val_ds, test_ds = cargar_datasets()

# Data augmentation DENTRO DEL PIPELINE de datos (tf.data) y no dentro del modelo.
# Lo medimos: con el backend PyTorch, rotar/zoom/desplazar dentro del modelo tardaba
# ~390 ms por lote y la GPU pasaba el 75 % del tiempo esperando. En tf.data las mismas
# capas de Keras se ejecutan con TensorFlow en la CPU, en paralelo (num_parallel_calls),
# mientras la GPU entrena el lote anterior (prefetch). Mismo efecto, mucho más rápido.
# Va DESPUÉS del cache() (dentro de cargar_datasets), así cada época ve variaciones nuevas.
import tensorflow as tf

aumento = capa_aumento_datos()


def aumentar(x, y):
    # Se aplican las capas una por una: cada capa de aumento sabe ejecutarse dentro de
    # tf.data, pero el contenedor Sequential no (con el backend PyTorch).
    for capa in aumento.layers:
        x = capa(x, training=True)
    return x, y


train_ds = train_ds.map(aumentar, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


# 2. Arquitectura ----------------------------------------------------------------
def bloque_conv(x, filtros, nombre):
    """Conv-BN-ReLU -> Conv-BN-ReLU -> MaxPool."""
    for i in (1, 2):
        # use_bias=False: la BatchNormalization ya tiene su propio desplazamiento (beta)
        x = layers.Conv2D(filtros, 3, padding="same", use_bias=False, name=f"{nombre}_conv{i}")(x)
        x = layers.BatchNormalization(name=f"{nombre}_bn{i}")(x)
        x = layers.Activation("relu", name=f"{nombre}_relu{i}")(x)
    return layers.MaxPooling2D(name=f"{nombre}_pool")(x)


entrada = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="imagen_rgb_0_255")
# (el aumento de datos ya se aplicó en train_ds, ver arriba)
x = layers.Rescaling(1.0 / 255, name="normalizar_0_1")(entrada)
for n, f in enumerate(FILTROS, start=1):            # 160 -> 80 -> 40 -> 20 -> 10
    x = bloque_conv(x, f, f"bloque{n}")
x = layers.GlobalAveragePooling2D(name="gap")(x)    # 10x10x256 -> vector de 256
x = layers.Dropout(0.4)(x)
salida = layers.Dense(len(CLASES), activation="softmax", name="clasificador")(x)
modelo = keras.Model(entrada, salida, name=NOMBRE)

modelo.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
modelo.summary()

# 3. Callbacks -------------------------------------------------------------------
callbacks = [
    keras.callbacks.ModelCheckpoint(DIR_MODELOS / f"{NOMBRE}.keras",
                                    monitor="val_accuracy", save_best_only=True, verbose=1),
    # Mucha más paciencia que en el experimento E2 (8): desde cero se aprende lento y val es ruidosa
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6,
                                      min_lr=1e-5, verbose=1),
]

# 4. Entrenamiento ---------------------------------------------------------------
t0 = time.time()
history = modelo.fit(train_ds, validation_data=val_ds, epochs=EPOCAS,
                     callbacks=callbacks, verbose=2)
minutos = (time.time() - t0) / 60

# 5. Guardar resultados ----------------------------------------------------------
guardar_historial(history.history, NOMBRE)
mejor = keras.models.load_model(DIR_MODELOS / f"{NOMBRE}.keras")
test_loss, test_acc = mejor.evaluate(test_ds, verbose=0)
h = history.history
guardar_metadatos(NOMBRE, {
    "tipo": "CNN desde cero v2 (BatchNorm + 2 conv por bloque + GlobalAveragePooling)",
    "test_accuracy": round(float(test_acc), 4),
    "mejor_val_accuracy": round(max(h["val_accuracy"]), 4),
    "epocas_entrenadas": len(h["loss"]),
    "minutos_entrenamiento": round(minutos, 1),
    "parametros": int(mejor.count_params()),
    "entrenado_en": dispositivo,
    "dataset": os.environ.get("DATASET", "dataset_cnn"),
    "semilla_entrenamiento": SEMILLA_ENTRENAMIENTO,
})
print(f"\nÉpocas: {len(h['loss'])}  |  Tiempo: {minutos:.1f} min en {dispositivo}")
print(f"Mejor val_accuracy: {max(h['val_accuracy']):.4f}")
print(f"Accuracy en TEST (imágenes nunca vistas): {test_acc:.4f}")
