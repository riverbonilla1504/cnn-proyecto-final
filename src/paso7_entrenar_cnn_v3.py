"""
PASO 7 - CNN v3 desde cero: animales LEJOS, frame COMPLETO y varios animales a la vez.

Qué cambia respecto a la v2 (paso 3) y por qué:

  1. Datos (paso 6): recortes con contexto (el animal ocupa del 2 % al 100 % de la imagen),
     cámaras trampa de día y de noche, escenas vacías, personas y vehículos como fondo.

  2. Resolución 320 px en entrenamiento (antes 224) y red TOTALMENTE CONVOLUCIONAL:
     no tiene capas que fijen el tamaño de entrada, así que en video se le puede pasar el
     frame entero a 640x360 (o el tamaño que se quiera) sin aplastarlo a 224x224.
     Un animal a 10 m ocupa ~30 px en un frame de 640: la red lo sigue "viendo".

  3. Un bloque convolucional más (5 en vez de 4): con 320-640 px de entrada hace falta un
     campo receptivo más grande (cada neurona del final "ve" ~200 px de la imagen).
     Para que entrenar no tarde el doble: la primera convolución avanza de 2 en 2 píxeles
     (stride 2, como en ResNet) y el 60 % de las épocas se entrena a 256 px y el resto a
     320 px (redimensionado progresivo). Medido en la RTX 5070: 7.1 -> 3.9 ms por imagen.

  4. MAPA DE EVIDENCIAS en vez de GlobalAveragePooling + Dense:
       - Una convolución 1x1 da, para CADA zona de la imagen, un puntaje por animal.
       - El puntaje de la imagen es un "máximo suave" de las zonas (LogSumExp, r = 5).
         Con el promedio (GAP) un animal pequeño se diluía entre todo el fondo; con el
         máximo basta con que UNA zona tenga una vaca para decir "hay vaca".
       - De regalo, el mapa dice DÓNDE está el animal (se dibuja la caja en el video).
     (Idea de Pinheiro y Collobert 2015 y de los "class activation maps", Zhou et al. 2016.)

  5. MULTI-ETIQUETA (4 sigmoides: bovino, caballo, gato, perro) en vez de softmax de 5:
     una escena puede tener vaca Y perro. "fondo" = ningún animal supera el umbral.
     Pérdida: entropía cruzada binaria con más peso a las clases con menos ejemplos.

  6. Aumento "de cámara de seguridad" además del de siempre: baja resolución y vuelta a
     ampliar, JPEG malo, ruido, gris con contraste alto (como el infrarrojo de noche).

Se mantiene: pesos aleatorios (nada preentrenado), AdamW, calentamiento + coseno,
precisión mixta en la GPU, guardado en float32 para CPU.

Uso:   .venv\\Scripts\\python.exe src\\paso7_entrenar_cnn_v3.py
       ($env:SEMILLA = "1" para otra semilla; $env:EPOCAS = "2" para una prueba corta)
Salida: modelos/cnn_v3[_sN].keras + .json, resultados/curvas_cnn_v3*.png, historial_*.json
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import json
import math
import time

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers

from capas import PoolingLSE
from comun import DIR_MODELOS, DIR_RESULTADOS, RAIZ, SEED, aumento_rapido, guardar_historial

ANIMALES = ["bovino", "caballo", "gato", "perro"]
DIR_DATOS = RAIZ / "data" / "dataset_v3"
IMG = 320
IMG_INICIAL = 256                      # redimensionado progresivo: el 60 % de las épocas a 256 px
FRACCION_INICIAL = 0.6
VAL_ANCHO, VAL_ALTO = 480, 360         # validación durante el fit (fotos completas, mismo tamaño para el lote)
BATCH = 48
EPOCAS = int(os.environ.get("EPOCAS", 25))
EPOCAS_CALENTAMIENTO = min(3, max(1, EPOCAS // 10))
LR_MAX = 2e-3
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05
FILTROS = [32, 64, 128, 256, 320]
R_LSE = 5.0

SEMILLA = int(os.environ.get("SEMILLA", SEED))
NOMBRE = "cnn_v3" + (f"_s{SEMILLA}" if SEMILLA != SEED else "")

keras.utils.set_random_seed(SEMILLA)
dispositivo = "CPU"
if keras.backend.backend() == "torch":
    import torch
    if torch.cuda.is_available():
        dispositivo = f"GPU ({torch.cuda.get_device_name(0)})"
        torch.backends.cudnn.benchmark = True
        keras.mixed_precision.set_global_policy("mixed_float16")
print(f"Backend: {keras.backend.backend()} | {dispositivo} | modelo: {NOMBRE}")

# 1. Datos ---------------------------------------------------------------------------------
man = pd.read_csv(os.environ.get("MANIFEST", DIR_DATOS / "manifest.csv"))   # MANIFEST: solo para pruebas
tr, va = man[man.split == "train"], man[man.split == "val"]
pos = tr[ANIMALES].sum().values.astype("float32")
neg = len(tr) - pos
peso_pos = np.clip(np.sqrt(neg / pos), 1.0, 4.0).astype("float32")   # clase escasa -> más peso
print(f"Train: {len(tr)} imágenes | val: {len(va)} fotos completas")
print("Positivos por clase:", dict(zip(ANIMALES, pos.astype(int))), "| fondo:", int((tr[ANIMALES].sum(1) == 0).sum()))
print("Peso de los positivos:", dict(zip(ANIMALES, peso_pos.round(2))))


def leer(ruta, y):
    x = tf.io.decode_jpeg(tf.io.read_file(ruta), channels=3)
    return x, y


def degradar_como_camara(x, y, IMG=IMG):
    """Una imagen (uint8) -> recorte aleatorio a IMG px + defectos de cámara de seguridad."""
    lado = tf.cast(tf.cast(tf.shape(x)[0], tf.float32) * tf.random.uniform([], 0.75, 1.0), tf.int32)
    x = tf.image.random_crop(x, [lado, lado, 3])
    x = tf.image.resize(x, [IMG, IMG])                                    # float32 0-255
    # baja resolución (cámara barata o animal lejos) y vuelta a ampliar
    f = tf.random.uniform([], 0.25, 0.8)
    chico = tf.cast(tf.cast(IMG, tf.float32) * f, tf.int32)
    x = tf.cond(tf.random.uniform([]) < 0.4,
                lambda: tf.image.resize(tf.image.resize(x, [chico, chico]), [IMG, IMG]), lambda: x)
    # compresión JPEG fuerte (bloques y bordes sucios del video)
    x = tf.cond(tf.random.uniform([]) < 0.3,
                lambda: tf.image.adjust_jpeg_quality(x / 255.0, tf.random.uniform([], 15, 60, tf.int32)) * 255.0,
                lambda: x)
    # infrarrojo nocturno: gris con más contraste
    def ir():
        g = tf.image.rgb_to_grayscale(x)
        g = (g - tf.reduce_mean(g)) * tf.random.uniform([], 1.1, 1.6) + tf.reduce_mean(g)
        return tf.image.grayscale_to_rgb(g)
    x = tf.cond(tf.random.uniform([]) < 0.12, ir, lambda: x)
    # ruido del sensor
    x = tf.cond(tf.random.uniform([]) < 0.3,
                lambda: x + tf.random.normal(tf.shape(x), 0.0, tf.random.uniform([], 2.0, 10.0)), lambda: x)
    return tf.clip_by_value(x, 0.0, 255.0), y


def lote_aumentado(x, y):
    # geometría + color en lote (comun.aumento_rapido); rotación más suave que en v2
    return aumento_rapido(x, rot=10.0, zoom=(0.85, 1.2), desplaz=0.08, shear=0.05), y


def frame_val(ruta, y):
    x = tf.io.decode_jpeg(tf.io.read_file(ruta), channels=3)
    return tf.image.resize(x, [VAL_ALTO, VAL_ANCHO]), y


AUTO = tf.data.AUTOTUNE
rutas_tr = [str(DIR_DATOS / a) for a in tr.archivo]


def datos_train(img):
    return (tf.data.Dataset.from_tensor_slices((rutas_tr, tr[ANIMALES].values.astype("float32")))
            .shuffle(len(tr), seed=SEMILLA, reshuffle_each_iteration=True)
            .map(leer, AUTO).map(lambda x, y: degradar_como_camara(x, y, img), AUTO)
            .batch(BATCH, drop_remainder=True).map(lote_aumentado, AUTO).prefetch(AUTO))


val_ds = (tf.data.Dataset.from_tensor_slices(([str(DIR_DATOS / a) for a in va.archivo],
                                              va[ANIMALES].values.astype("float32")))
          .map(frame_val, AUTO).batch(32).cache().prefetch(AUTO))
pasos_por_epoca = len(tr) // BATCH


# 2. Arquitectura -------------------------------------------------------------------------
def construir_modelo():
    def bloque_conv(x, filtros, nombre, entrada=False):
        for i in (1, 2):
            # bloque 1: la PRIMERA convolución avanza de 2 en 2 píxeles (stride 2) en vez de
            # usar MaxPooling al final. Medido: entrenar es 1.8x más rápido (la convolución a
            # resolución completa era la más cara) y la red sigue reduciendo 32x en total.
            paso = 2 if (entrada and i == 1) else 1
            x = layers.Conv2D(filtros, 3, strides=paso, padding="same", use_bias=False, name=f"{nombre}_conv{i}")(x)
            x = layers.BatchNormalization(name=f"{nombre}_bn{i}")(x)
            x = layers.Activation("relu", name=f"{nombre}_relu{i}")(x)
        return x if entrada else layers.MaxPooling2D(name=f"{nombre}_pool")(x)

    entrada = layers.Input(shape=(None, None, 3), name="imagen_rgb_0_255")   # cualquier tamaño
    x = layers.Rescaling(1.0 / 255, name="normalizar_0_1")(entrada)
    for n, f in enumerate(FILTROS, start=1):            # 320 -> 160 -> 80 -> 40 -> 20 -> 10
        x = bloque_conv(x, f, f"bloque{n}", entrada=(n == 1))
    x = layers.SpatialDropout2D(0.2, name="dropout_zonas")(x)
    mapa = layers.Conv2D(len(ANIMALES), 1, name="mapa_evidencias", dtype="float32")(x)   # (N, h, w, 4)
    puntaje = PoolingLSE(R_LSE, name="max_suave", dtype="float32")(mapa)
    salida = layers.Activation("sigmoid", name="probabilidades", dtype="float32")(puntaje)
    return keras.Model(entrada, salida, name=NOMBRE)


modelo = construir_modelo()
pp = keras.ops.convert_to_tensor(peso_pos)


def bce_pesada(y, p):
    """Entropía cruzada binaria; los positivos de clases escasas pesan más."""
    y = y * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
    p = keras.ops.clip(p, 1e-6, 1 - 1e-6)
    l = -(pp * y * keras.ops.log(p) + (1 - y) * keras.ops.log(1 - p))
    return keras.ops.mean(l, axis=-1)


# 3. Optimizador ---------------------------------------------------------------------------
lr = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LR_MAX / 100, warmup_target=LR_MAX,
    warmup_steps=EPOCAS_CALENTAMIENTO * pasos_por_epoca,
    decay_steps=(EPOCAS - EPOCAS_CALENTAMIENTO) * pasos_por_epoca, alpha=0.01)
opt = keras.optimizers.AdamW(learning_rate=lr, weight_decay=WEIGHT_DECAY)
opt.exclude_from_weight_decay(var_names=["gamma", "beta", "bias"])
metricas = [keras.metrics.AUC(curve="PR", multi_label=True, num_labels=len(ANIMALES), name="map"),
            keras.metrics.BinaryAccuracy(threshold=0.5, name="acc_binaria")]
modelo.compile(optimizer=opt, loss=bce_pesada, metrics=metricas)
modelo.summary()

ruta_tmp = DIR_MODELOS / f"_{NOMBRE}_entrenando.keras"
callbacks = [keras.callbacks.ModelCheckpoint(ruta_tmp, monitor="val_map", mode="max", save_best_only=True, verbose=1),
             keras.callbacks.CSVLogger(str(DIR_RESULTADOS / f"epocas_{NOMBRE}.csv"), append=True)]
(DIR_RESULTADOS / f"epocas_{NOMBRE}.csv").unlink(missing_ok=True)

# 4. Entrenamiento -------------------------------------------------------------------------
# Redimensionado progresivo (idea de EfficientNetV2, Tan y Le 2021, y de FixRes, Touvron 2019):
# las primeras épocas, cuando la red aprende formas gruesas, a 256 px (~1.6x más rápido);
# las últimas a 320 px para afinar los detalles. Como la red es totalmente convolucional,
# es el MISMO modelo: solo cambia el tamaño de las imágenes que recibe.
t0 = time.time()
corte = int(round(EPOCAS * FRACCION_INICIAL)) if EPOCAS >= 5 else 0
historias = []
if corte:
    print(f"\n== Fase 1: épocas 1-{corte} a {IMG_INICIAL} px ==")
    historias.append(modelo.fit(datos_train(IMG_INICIAL), validation_data=val_ds, epochs=corte,
                                callbacks=callbacks, verbose=2).history)
print(f"\n== Fase 2: épocas {corte + 1}-{EPOCAS} a {IMG} px ==")
historias.append(modelo.fit(datos_train(IMG), validation_data=val_ds, initial_epoch=corte, epochs=EPOCAS,
                            callbacks=callbacks, verbose=2).history)
minutos = (time.time() - t0) / 60

# 5. Guardar en float32 ----------------------------------------------------------------------
mejor = keras.models.load_model(ruta_tmp, custom_objects={"bce_pesada": bce_pesada})
keras.mixed_precision.set_global_policy("float32")
final = construir_modelo()
final.set_weights(mejor.get_weights())
final.save(DIR_MODELOS / f"{NOMBRE}.keras")
ruta_tmp.unlink()

h = {k: sum((hh.get(k, []) for hh in historias), []) for k in historias[-1]}
h["accuracy"], h["val_accuracy"] = h["map"], h["val_map"]          # para reutilizar las curvas
guardar_historial(h, NOMBRE)
meta = {
    "clases": ANIMALES, "tipo_salida": "multi-etiqueta (sigmoide por animal); fondo = ninguno supera el umbral",
    "img_size_entrenamiento": IMG, "entrada": "imagen RGB 0-255 de CUALQUIER tamaño (red totalmente convolucional)",
    "tamano_recomendado_video": [640, 360], "arquitectura": f"5 bloques {FILTROS} + mapa 1x1 + LogSumExp r={R_LSE}",
    "mejor_val_map": round(max(h["val_map"]), 4), "epocas": len(h["loss"]),
    "minutos_entrenamiento": round(minutos, 1), "parametros": int(final.count_params()),
    "entrenado_en": dispositivo, "semilla_entrenamiento": SEMILLA,
    "positivos_train": dict(zip(ANIMALES, pos.astype(int).tolist())),
}
(DIR_MODELOS / f"{NOMBRE}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nÉpocas: {len(h['loss'])} | {minutos:.1f} min en {dispositivo} | mejor val mAP: {max(h['val_map']):.4f}")
