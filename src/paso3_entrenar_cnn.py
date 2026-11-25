"""
PASO 3 - CNN desde cero v2 con RECETA MODERNA de entrenamiento, a 224 px, en GPU.

La ARQUITECTURA es la misma de la v2 (experimento previo, historial/src/exp3_entrenar_cnn_v2.py): 4 bloques Conv-BN-ReLU x2 + MaxPool,
GlobalAveragePooling, Dropout y Dense(5). Nada preentrenado: pesos aleatorios.
Lo que cambia es CÓMO se entrena (ideas de "Bag of Tricks", He et al. 2019, y de
"ResNet strikes back", Wightman et al. 2021):

  1. Resolución 224x224 (antes 160): los recortes miden ~370 px de mediana; a 160 se
     perdían detalles (orejas, hocico, cuernos). Cuesta ~2x de cómputo.
  2. AdamW (Adam + weight decay desacoplado): castiga pesos grandes -> menos sobreajuste.
     No se aplica a BatchNorm ni a los sesgos (práctica estándar).
  3. Learning rate con CALENTAMIENTO (warm-up) de 5 épocas y luego DECAIMIENTO COSENO:
     empieza bajo (la red con pesos al azar es inestable), sube al máximo y baja suave
     hasta casi 0. Reemplaza a ReduceLROnPlateau; el número de épocas es fijo.
  4. Label smoothing 0.1: la etiqueta "perro" pasa de [0,0,0,0,1] a [.02,.02,.02,.02,.92].
     El modelo no se vuelve "sobreconfiado" y tolera mejor las etiquetas mal puestas
     (vimos ~1-2 % en Open Images).
  5. Aumento de datos fuerte (en tf.data, CPU, en paralelo):
       - Una transformación afín aleatoria por foto: volteo, rotación +-15 grados, zoom,
         desplazamiento e inclinación (antes eran 4 capas separadas).
       - Color aleatorio por foto: brillo, contraste, saturación, 10 % en escala de grises.
         (Misma idea que RandAugment + color jitter; ver comun.aumento_rapido: las capas de
         Keras eran 14x más lentas y dejaban la GPU esperando.)
       - MixUp o CutMix (50 % de los lotes, uno de los dos al azar):
           MixUp mezcla dos fotos (70 % perro + 30 % vaca) y también sus etiquetas.
           CutMix pega un rectángulo de otra foto y mezcla las etiquetas según el área.
         Obligan a la red a usar TODA la imagen y no un solo detalle.
  6. Pesos por clase: bovino tiene ~3 300 fotos y las demás ~6 000; cada foto de bovino
     "pesa" más en la pérdida para que la red no lo descuide.
  7. Precisión mixta (float16 en la GPU): ~2x más rápido. Al final el modelo se guarda
     en float32 para que funcione igual en CPU (clasificador.py / cámara).

Salidas:  modelos/cnn_propia_v2_receta[_sN].keras + .json, resultados/curvas_*.png
Uso (entrena con data/dataset_cnn_maximo, el dataset del paso 2):
    .venv\\Scripts\\python.exe src\\paso3_entrenar_cnn.py
    ($env:SEMILLA = "1" y luego "2" para repetir con otras semillas; las 3 van al paso 4)
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import time

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from comun import (CLASES, DIR_MODELOS, DIR_TRAIN, SEED, aumento_rapido, cargar_datasets,
                   guardar_historial, guardar_metadatos)

IMG = 224
BATCH = 64
EPOCAS = int(os.environ.get("EPOCAS", 60))
EPOCAS_CALENTAMIENTO = min(5, max(1, EPOCAS // 10))   # 5 con 60 épocas
LR_MAX = 2e-3
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
PROB_MEZCLA = 0.5          # fracción de lotes con MixUp o CutMix
FILTROS = [32, 64, 128, 256]

NOMBRE = "cnn_propia_v2_receta"
if os.environ.get("DATASET", "dataset_cnn_maximo") != "dataset_cnn_maximo":
    NOMBRE += "_" + os.environ["DATASET"].removeprefix("dataset_cnn").strip("_")
SEMILLA = int(os.environ.get("SEMILLA", SEED))
if SEMILLA != SEED:
    NOMBRE += f"_s{SEMILLA}"
NOMBRE = NOMBRE.rstrip("_")

keras.utils.set_random_seed(SEMILLA)
dispositivo = "CPU"
if keras.backend.backend() == "torch":
    import torch
    if torch.cuda.is_available():
        dispositivo = f"GPU ({torch.cuda.get_device_name(0)})"
        torch.backends.cudnn.benchmark = True       # elige el algoritmo de convolución más rápido
        keras.mixed_precision.set_global_policy("mixed_float16")
print(f"Backend: {keras.backend.backend()} | {dispositivo} | modelo: {NOMBRE}")

# 1. Datos -----------------------------------------------------------------------------
train_ds, val_ds, test_ds = cargar_datasets(IMG, BATCH, one_hot=True)

conteo = np.array([len(list((DIR_TRAIN / c).glob("*.jpg"))) for c in CLASES], dtype="float32")
peso_clase = conteo.sum() / (len(CLASES) * conteo)          # clase escasa -> peso mayor
print("Fotos por clase:", dict(zip(CLASES, conteo.astype(int))),
      "| pesos:", dict(zip(CLASES, peso_clase.round(2))))
pesos_tf = tf.constant(peso_clase)

# dtype="float32": el aumento corre en la CPU (tf.data); la precisión mixta es solo para la GPU
F32 = dict(dtype="float32")
mixup = layers.MixUp(alpha=0.2, **F32)
cutmix = layers.CutMix(factor=1.0, **F32)


def aumentar(x, y):
    # Geometría + color en una sola pasada (comun.aumento_rapido). Primero probamos
    # RandAugment y RandomColorJitter de Keras: tardaban 2.8 s por lote y dejaban la GPU
    # esperando (1 s por paso). aumento_rapido hace lo mismo en ~0.2 s.
    x = aumento_rapido(x)
    r = tf.random.uniform(())

    def con_mixup():
        d = mixup({"images": x, "labels": y}, training=True)
        return d["images"], d["labels"]

    def con_cutmix():
        d = cutmix({"images": x, "labels": y}, training=True)
        return d["images"], d["labels"]

    x, y = tf.case([(r < PROB_MEZCLA / 2, con_mixup), (r < PROB_MEZCLA, con_cutmix)],
                   default=lambda: (x, y))
    # peso de cada foto según su clase (si la etiqueta está mezclada, se mezcla el peso)
    w = tf.reduce_sum(y * pesos_tf, axis=-1)
    return x, y, w


train_ds = train_ds.map(aumentar, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
pasos_por_epoca = int(train_ds.cardinality())


# 2. Arquitectura (idéntica a la v2) -----------------------------------------------------
def construir_modelo():
    def bloque_conv(x, filtros, nombre):
        for i in (1, 2):
            x = layers.Conv2D(filtros, 3, padding="same", use_bias=False, name=f"{nombre}_conv{i}")(x)
            x = layers.BatchNormalization(name=f"{nombre}_bn{i}")(x)
            x = layers.Activation("relu", name=f"{nombre}_relu{i}")(x)
        return layers.MaxPooling2D(name=f"{nombre}_pool")(x)

    entrada = layers.Input(shape=(IMG, IMG, 3), name="imagen_rgb_0_255")
    x = layers.Rescaling(1.0 / 255, name="normalizar_0_1")(entrada)
    for n, f in enumerate(FILTROS, start=1):          # 224 -> 112 -> 56 -> 28 -> 14
        x = bloque_conv(x, f, f"bloque{n}")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4)(x)
    # la salida siempre en float32 (softmax en float16 puede perder precisión)
    salida = layers.Dense(len(CLASES), activation="softmax", name="clasificador", dtype="float32")(x)
    return keras.Model(entrada, salida, name=NOMBRE)


modelo = construir_modelo()

# 3. Optimizador: AdamW + calentamiento + coseno ------------------------------------------
lr = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LR_MAX / 100,                 # empieza en 2e-5
    warmup_target=LR_MAX, warmup_steps=EPOCAS_CALENTAMIENTO * pasos_por_epoca,
    decay_steps=(EPOCAS - EPOCAS_CALENTAMIENTO) * pasos_por_epoca, alpha=0.01)
opt = keras.optimizers.AdamW(learning_rate=lr, weight_decay=WEIGHT_DECAY)
opt.exclude_from_weight_decay(var_names=["gamma", "beta", "bias"])
modelo.compile(optimizer=opt,
               loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
               metrics=["accuracy"], weighted_metrics=[])
modelo.summary()

ruta_tmp = DIR_MODELOS / f"_{NOMBRE}_entrenando.keras"
callbacks = [keras.callbacks.ModelCheckpoint(ruta_tmp, monitor="val_accuracy",
                                             save_best_only=True, verbose=1)]

# 4. Entrenamiento ------------------------------------------------------------------------
t0 = time.time()
history = modelo.fit(train_ds, validation_data=val_ds, epochs=EPOCAS, callbacks=callbacks, verbose=2)
minutos = (time.time() - t0) / 60

# 5. Guardar en float32 (para CPU / cámara) y evaluar --------------------------------------
mejor = keras.models.load_model(ruta_tmp)
keras.mixed_precision.set_global_policy("float32")
final = construir_modelo()
final.set_weights(mejor.get_weights())
final.save(DIR_MODELOS / f"{NOMBRE}.keras")         # se guarda SIN optimizador (solo para predecir)
final.compile(loss="categorical_crossentropy", metrics=["accuracy"])
ruta_tmp.unlink()

guardar_historial(history.history, NOMBRE)
test_loss, test_acc = final.evaluate(test_ds, verbose=0)
h = history.history
guardar_metadatos(NOMBRE, {
    "img_size": IMG,
    "tipo": "CNN desde cero v2 + receta moderna (AdamW, coseno, label smoothing, RandAugment, MixUp/CutMix)",
    "test_accuracy": round(float(test_acc), 4),
    "mejor_val_accuracy": round(max(h["val_accuracy"]), 4),
    "epocas_entrenadas": len(h["loss"]),
    "minutos_entrenamiento": round(minutos, 1),
    "parametros": int(final.count_params()),
    "entrenado_en": dispositivo,
    "dataset": os.environ.get("DATASET", "dataset_cnn_maximo"),
    "semilla_entrenamiento": SEMILLA,
    "fotos_por_clase": dict(zip(CLASES, conteo.astype(int).tolist())),
})
print(f"\nÉpocas: {len(h['loss'])}  |  Tiempo: {minutos:.1f} min en {dispositivo}")
print(f"Mejor val_accuracy: {max(h['val_accuracy']):.4f}")
print(f"Accuracy en TEST (imágenes nunca vistas): {test_acc:.4f}")
