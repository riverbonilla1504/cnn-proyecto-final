"""
EXPERIMENTO E2 - Entrenar una CNN DESDE CERO (modelo base / "baseline").

Es la misma idea del ejemplo de clase con MNIST (Conv2D -> MaxPooling -> Flatten -> Dense),
pero adaptada a fotos a color de 160x160 y 5 clases.

Es la PRIMERA versión (v1). Sirve como punto de partida: con sus resultados y sus
curvas decidimos qué mejorar en la versión 2 (experimento E3).

Salidas:
  modelos/cnn_propia.keras        (mejor modelo según validación)
  modelos/cnn_propia.json         (clases y tamaño de entrada)
  resultados/curvas_cnn_propia.png
  resultados/historial_cnn_propia.json

Uso:  .venv\\Scripts\\python.exe historial\\src\\exp2_entrenar_cnn_v1.py
"""
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from comun import (CLASES, DIR_MODELOS, IMG_SIZE, SEED, capa_aumento_datos,
                   cargar_datasets, guardar_historial, guardar_metadatos)

NOMBRE = "cnn_propia"
EPOCAS = 40               # máximo; EarlyStopping puede parar antes
LEARNING_RATE = 1e-3

keras.utils.set_random_seed(SEED)

# 1. Datos -----------------------------------------------------------------------
train_ds, val_ds, test_ds = cargar_datasets()

# 2. Arquitectura ----------------------------------------------------------------
# Cada bloque: Conv2D (aprende filtros/patrones) + MaxPooling (reduce el tamaño a la mitad).
# Las primeras capas aprenden bordes y colores; las siguientes, texturas y formas
# (orejas, patas, cuernos...). Al final, capas densas deciden la clase.
modelo = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),      # imagen RGB 160x160
    capa_aumento_datos(),                             # solo actúa al entrenar
    layers.Rescaling(1.0 / 255),                      # píxeles 0-255 -> 0-1 (normalización)

    layers.Conv2D(32, 3, padding="same", activation="relu"),   # 160x160x32
    layers.MaxPooling2D(),                                      # 80x80x32
    layers.Conv2D(64, 3, padding="same", activation="relu"),   # 80x80x64
    layers.MaxPooling2D(),                                      # 40x40x64
    layers.Conv2D(128, 3, padding="same", activation="relu"),  # 40x40x128
    layers.MaxPooling2D(),                                      # 20x20x128
    layers.Conv2D(128, 3, padding="same", activation="relu"),  # 20x20x128
    layers.MaxPooling2D(),                                      # 10x10x128

    layers.Flatten(),                                 # 10*10*128 = 12 800 valores
    layers.Dropout(0.5),                              # apaga 50 % de neuronas al azar -> menos sobreajuste
    layers.Dense(128, activation="relu"),
    layers.Dense(len(CLASES), activation="softmax"),  # 5 probabilidades que suman 1
], name=NOMBRE)

modelo.compile(
    optimizer=keras.optimizers.Adam(LEARNING_RATE),
    loss="sparse_categorical_crossentropy",   # etiquetas enteras (0..4)
    metrics=["accuracy"],
)
modelo.summary()

# 3. Callbacks -------------------------------------------------------------------
callbacks = [
    # Guarda el modelo solo cuando mejora la accuracy de validación
    keras.callbacks.ModelCheckpoint(DIR_MODELOS / f"{NOMBRE}.keras",
                                    monitor="val_accuracy", save_best_only=True, verbose=1),
    # Si val_loss no mejora en 8 épocas, para y recupera los mejores pesos
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    # Si val_loss se estanca 4 épocas, baja el learning rate a la mitad
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
]

# 4. Entrenamiento ---------------------------------------------------------------
t0 = time.time()
history = modelo.fit(train_ds, validation_data=val_ds, epochs=EPOCAS, callbacks=callbacks)
minutos = (time.time() - t0) / 60

# 5. Guardar resultados ----------------------------------------------------------
guardar_historial(history.history, NOMBRE)
mejor = keras.models.load_model(DIR_MODELOS / f"{NOMBRE}.keras")
test_loss, test_acc = mejor.evaluate(test_ds, verbose=0)
guardar_metadatos(NOMBRE, {"tipo": "CNN desde cero", "test_accuracy": round(test_acc, 4),
                           "epocas_entrenadas": len(history.history["loss"]),
                           "minutos_entrenamiento": round(minutos, 1),
                           "parametros": int(mejor.count_params())})
print(f"\nTiempo de entrenamiento: {minutos:.1f} min")
print(f"Accuracy en TEST (imágenes nunca vistas): {test_acc:.4f}")
