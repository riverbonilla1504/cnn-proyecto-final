"""
comun.py (COPIA ARCHIVADA para los experimentos previos E1-E4).
Es el comun.py tal como estaba cuando se corrieron. El pipeline final usa src/comun.py.


Tener todo aquí evita que cada script use rutas o tamaños de imagen distintos.
Si cambias IMG_SIZE o las clases, cambia solo aquí.
"""
import json
import math
import os
import sys
from pathlib import Path

# Silenciar mensajes informativos de TensorFlow (solo mostrar errores)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Para que la consola de Windows muestre bien tildes y eñes
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent          # carpeta historial/
RAIZ_PROYECTO = RAIZ.parent                            # carpeta del proyecto
# Por defecto el dataset original. Para usar el ampliado (paso 4):  $env:DATASET = "dataset_cnn_ampliado"
_nombre_ds = os.environ.get("DATASET", "dataset_cnn")
DIR_DATASET = (RAIZ / "data" / _nombre_ds if (RAIZ / "data" / _nombre_ds).exists()
               else RAIZ_PROYECTO / "data" / _nombre_ds)
DIR_TRAIN = DIR_DATASET / "train"                       # 2500 imágenes (500 por clase)
DIR_TEST = DIR_DATASET / "val"                          # 500 imágenes -> las usamos como TEST final
DIR_MODELOS = RAIZ / "modelos"
DIR_RESULTADOS = RAIZ / "resultados"

DIR_MODELOS.mkdir(exist_ok=True)
DIR_RESULTADOS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# HIPERPARÁMETROS GENERALES
# ---------------------------------------------------------------------------
IMG_SIZE = 160          # todas las imágenes se redimensionan a 160x160 píxeles
BATCH_SIZE = 32         # imágenes procesadas antes de actualizar los pesos
SEED = 42               # semilla para que los resultados sean reproducibles
VAL_SPLIT = 0.15        # 15 % del train se separa para validación

# Orden alfabético = el mismo orden que usa Keras al leer las carpetas
CLASES = ["bovino", "caballo", "fondo", "gato", "perro"]


def cargar_datasets():
    """
    Lee las imágenes desde las carpetas y devuelve (train, validacion, test).

    - Keras asigna la etiqueta según el nombre de la carpeta (bovino=0, caballo=1, ...).
    - train/validacion salen de la carpeta train (85 % / 15 %).
    - test es la carpeta val original: el modelo NUNCA la ve durante el entrenamiento.
    """
    import tensorflow as tf

    comunes = dict(
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",          # etiquetas como enteros 0..4 -> sparse_categorical_crossentropy
        class_names=CLASES,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DIR_TRAIN, validation_split=VAL_SPLIT, subset="training",
        seed=SEED, shuffle=True, **comunes)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DIR_TRAIN, validation_split=VAL_SPLIT, subset="validation",
        seed=SEED, shuffle=True, **comunes)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DIR_TEST, shuffle=False, **comunes)

    # cache + prefetch: guarda las imágenes en memoria y prepara el siguiente lote
    # mientras el modelo entrena el actual (solo velocidad, no cambia resultados)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds


def cargar_datasets_grandes(img_size, batch_size, one_hot=True):
    """
    Igual que cargar_datasets(), pero pensada para datasets grandes (paso 7):
      - Tamaño de imagen y de lote configurables (p. ej. 224 px y 64).
      - Etiquetas one-hot (necesarias para label smoothing y MixUp/CutMix).
      - Guarda en caché las imágenes como uint8 (0-255) en vez de float32: 4 veces menos RAM
        (27 000 fotos de 224x224 -> ~4 GB en vez de ~16 GB). Se convierten a float al usarlas.
      - Baraja FOTO por foto en cada época (no solo el orden de los lotes).
    Misma división que siempre: 85 % / 15 % de train con SEED=42; test = carpeta val.
    """
    import tensorflow as tf

    comunes = dict(image_size=(img_size, img_size), batch_size=None,
                   label_mode="categorical" if one_hot else "int", class_names=CLASES)
    a_uint8 = lambda x, y: (tf.cast(tf.round(x), tf.uint8), y)
    a_float = lambda x, y: (tf.cast(x, tf.float32), y)
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DIR_TRAIN, validation_split=VAL_SPLIT, subset="training", seed=SEED, shuffle=True, **comunes)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DIR_TRAIN, validation_split=VAL_SPLIT, subset="validation", seed=SEED, shuffle=True, **comunes)
    test_ds = tf.keras.utils.image_dataset_from_directory(DIR_TEST, shuffle=False, **comunes)

    train_ds = (train_ds.map(a_uint8, AUTOTUNE).cache()
                .shuffle(20000, seed=SEED, reshuffle_each_iteration=True)
                .batch(batch_size, drop_remainder=True).map(a_float, AUTOTUNE))
    val_ds = val_ds.map(a_uint8, AUTOTUNE).cache().batch(batch_size).map(a_float, AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(a_uint8, AUTOTUNE).cache().batch(batch_size).map(a_float, AUTOTUNE).prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds


def aumento_rapido(x, rot=15.0, zoom=(0.8, 1.25), desplaz=0.1, shear=0.1,
                   brillo=0.2, contraste=(0.7, 1.3), saturacion=(0.6, 1.4), p_gris=0.1):
    """
    Aumento de datos RÁPIDO para el paso 7 (misma idea que RandAugment + color jitter).
    x: lote (N, H, W, 3) float32 en 0-255. Todo se calcula para el lote completo a la vez.

    Por qué existe: medimos que RandAugment de Keras tardaba 2.4 s por lote de 64 fotos a
    224 px y el color jitter 0.4 s, mientras la GPU entrena un lote en 0.19 s: la GPU
    pasaba casi todo el tiempo esperando. Esta versión tarda ~0.2 s por lote (16x menos).

    Geometría: UNA sola transformación afín por imagen (volteo, rotación +-15 grados,
      zoom 0.8-1.25x, desplazamiento +-10 %, inclinación) en vez de 4 capas que
      deformaban la imagen 4 veces. Bordes rellenados en espejo (REFLECT).
    Color: saturación (10 % de las fotos queda en escala de grises), contraste y brillo,
      con un valor aleatorio distinto para cada foto.
    """
    import tensorflow as tf

    n = tf.shape(x)[0]
    h = tf.cast(tf.shape(x)[1], tf.float32)
    w = tf.cast(tf.shape(x)[2], tf.float32)
    # --- geometría: una sola transformación afín por imagen -------------------------
    ang = tf.random.uniform([n], -rot, rot) * math.pi / 180
    esc = tf.exp(tf.random.uniform([n], math.log(zoom[0]), math.log(zoom[1])))
    sh = tf.random.uniform([n], -shear, shear)
    tx = tf.random.uniform([n], -desplaz, desplaz) * w
    ty = tf.random.uniform([n], -desplaz, desplaz) * h
    flip = tf.where(tf.random.uniform([n]) < 0.5, -1.0, 1.0)
    cos, sin = tf.cos(ang), tf.sin(ang)
    # matriz salida->entrada (inversa): [a0 a1 a2; b0 b1 b2]
    a0 = cos / esc * flip
    a1 = (-sin + sh * cos) / esc
    b0 = sin / esc * flip
    b1 = (cos + sh * sin) / esc
    cx, cy = w / 2, h / 2
    a2 = cx - a0 * cx - a1 * cy + tx
    b2 = cy - b0 * cx - b1 * cy + ty
    ceros = tf.zeros([n])
    T = tf.stack([a0, a1, a2, b0, b1, b2, ceros, ceros], axis=1)
    x = tf.raw_ops.ImageProjectiveTransformV3(
        images=x, transforms=T, output_shape=tf.shape(x)[1:3], interpolation="BILINEAR",
        fill_mode="REFLECT", fill_value=0.0)
    # --- color: factores aleatorios por imagen -----------------------------------------
    gris = tf.reduce_mean(x, axis=-1, keepdims=True)
    s = tf.random.uniform([n, 1, 1, 1], *saturacion)
    s = tf.where(tf.random.uniform([n, 1, 1, 1]) < p_gris, 0.0, s)      # a veces gris total
    x = gris + (x - gris) * s
    media = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    x = (x - media) * tf.random.uniform([n, 1, 1, 1], *contraste) + media
    x = x + tf.random.uniform([n, 1, 1, 1], -brillo, brillo) * 255.0
    return tf.clip_by_value(x, 0.0, 255.0)


def capa_aumento_datos():
    """
    Data augmentation: pequeñas transformaciones aleatorias SOLO durante el entrenamiento.
    Así el modelo ve "versiones nuevas" de cada foto en cada época y memoriza menos.
    En predicción (inference) estas capas no hacen nada.
    """
    # "import keras" (Keras 3) funciona con cualquier backend: TensorFlow (CPU) o PyTorch (GPU)
    import keras
    from keras import layers

    return keras.Sequential([
        layers.RandomFlip("horizontal"),     # espejo izquierda-derecha (un perro sigue siendo perro)
        layers.RandomRotation(0.08),         # rota hasta ±8 % de 360° (≈ ±29°)
        layers.RandomZoom(0.15),             # acerca/aleja hasta 15 %
        layers.RandomTranslation(0.1, 0.1),  # desplaza hasta 10 %
        layers.RandomContrast(0.2),          # cambia el contraste (día nublado, sombra...)
    ], name="aumento_datos")


def guardar_metadatos(nombre_modelo, extra=None):
    """Guarda un JSON junto al modelo con lo necesario para usarlo después (clases, tamaño)."""
    meta = {"clases": CLASES, "img_size": IMG_SIZE,
            "entrada": "imagen RGB 0-255 sin normalizar (el modelo normaliza por dentro)"}
    if extra:
        meta.update(extra)
    ruta = DIR_MODELOS / f"{nombre_modelo}.json"
    ruta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta


def guardar_historial(history, nombre):
    """Guarda las curvas de entrenamiento (loss/accuracy por época) en JSON y PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = {k: [float(x) for x in v] for k, v in history.items()}
    (DIR_RESULTADOS / f"historial_{nombre}.json").write_text(json.dumps(h, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(h["accuracy"], label="Entrenamiento")
    ax[0].plot(h["val_accuracy"], label="Validación")
    ax[0].set_title("Accuracy por época"); ax[0].set_xlabel("Época"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(h["loss"], label="Entrenamiento")
    ax[1].plot(h["val_loss"], label="Validación")
    ax[1].set_title("Loss por época"); ax[1].set_xlabel("Época"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle(f"Curvas de entrenamiento - {nombre}")
    fig.tight_layout()
    fig.savefig(DIR_RESULTADOS / f"curvas_{nombre}.png", dpi=110)
    plt.close(fig)
