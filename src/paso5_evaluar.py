"""
PASO 5 - Evaluar el modelo final con el conjunto de TEST (500 imágenes nunca vistas).

Evalúa una red sola (semilla 42) y el modelo final. Para cada uno calcula:
  - Accuracy global
  - Precision, recall y F1 por clase (classification_report)
  - Matriz de confusión (qué clase confunde con cuál)
  - Mosaico con las fotos mal clasificadas, para analizarlas
Cada modelo se evalúa con SU tamaño de imagen (224 px, sale de su .json).
El "modelo final" se evalúa igual que lo usa la cámara: con clasificador.py
(si es un ensemble + TTA, promedia todos sus modelos y la imagen en espejo).

Salidas:
  resultados/matriz_confusion_<modelo>.png
  resultados/errores_<modelo>.png
  resultados/paso5_evaluacion.txt

Uso:  .venv\\Scripts\\python.exe src\\paso5_evaluar.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from clasificador import ClasificadorAnimales
from comun import CLASES, DIR_MODELOS, DIR_RESULTADOS, DIR_TEST

MODELOS = {
    "cnn_propia_v2_receta": "Una sola CNN (semilla 42), receta moderna, 224 px",
    "modelo_final": "MODELO FINAL (ensemble 3 semillas + TTA)",
}

_cache = {}


def test_de_tamano(img):
    """Test (500 fotos) redimensionado a img x img, igual que en el entrenamiento."""
    if img not in _cache:
        ds = tf.keras.utils.image_dataset_from_directory(
            DIR_TEST, image_size=(img, img), batch_size=64, shuffle=False,
            label_mode="int", class_names=CLASES)
        x = np.concatenate([a.numpy() for a, _ in ds]); y = np.concatenate([b.numpy() for _, b in ds])
        _cache[img] = (x, y)
    return _cache[img]


informe = []
for archivo, titulo in MODELOS.items():
    if not (DIR_MODELOS / f"{archivo}.json").exists():
        print(f"[!] Falta {archivo}.json, se omite")
        continue
    clf = ClasificadorAnimales(DIR_MODELOS / f"{archivo}.keras")
    x, y_real = test_de_tamano(clf.img_size)
    vistas = [x, x[:, :, ::-1]] if clf.tta else [x]
    probs = np.mean([m.predict(v, batch_size=64, verbose=0) for m in clf.modelos for v in vistas], axis=0)
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_real, y_pred)
    params = sum(m.count_params() for m in clf.modelos)

    detalle = f"{len(clf.modelos)} modelo(s){' + TTA' if clf.tta else ''}, {clf.img_size} px"
    bloque = [f"===== {titulo} ({archivo}) =====",
              f"{detalle} | parámetros: {params:,}",
              f"Accuracy en test: {acc:.4f}  ({int((y_pred == y_real).sum())}/{len(y_real)} correctas)",
              "",
              classification_report(y_real, y_pred, target_names=CLASES, digits=3)]
    print("\n".join(bloque))
    informe += bloque

    cm = confusion_matrix(y_real, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASES, yticklabels=CLASES, ax=ax)
    ax.set_xlabel("Predicción del modelo"); ax.set_ylabel("Clase real")
    ax.set_title(f"Matriz de confusión - {titulo}\naccuracy = {acc:.1%}")
    fig.tight_layout(); fig.savefig(DIR_RESULTADOS / f"matriz_confusion_{archivo}.png", dpi=110)
    plt.close(fig)

    errores = np.where(y_pred != y_real)[0][:36]
    if len(errores):
        cols = 6
        filas = int(np.ceil(len(errores) / cols))
        fig, axes = plt.subplots(filas, cols, figsize=(cols * 2.2, filas * 2.5))
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, i in zip(np.atleast_1d(axes).ravel(), errores):
            ax.imshow(x[i].astype("uint8"))
            ax.set_title(f"real: {CLASES[y_real[i]]}\npred: {CLASES[y_pred[i]]} {probs[i].max():.0%}",
                         fontsize=8, color="crimson")
        fig.suptitle(f"Errores en test - {titulo}")
        fig.tight_layout(); fig.savefig(DIR_RESULTADOS / f"errores_{archivo}.png", dpi=90)
        plt.close(fig)

# Repeticiones con distintas semillas: una sola corrida no basta para comparar,
# porque el azar del entrenamiento mueve el resultado varios puntos.
grupos = {"CNN sola, receta moderna": "cnn_propia_v2_receta"}
bloque = ["===== Repeticiones con distintas semillas (accuracy en test) ====="]
for etiqueta, base in grupos.items():
    accs = []
    for ruta in sorted(DIR_MODELOS.glob(f"{base}*.json")):
        sufijo = ruta.stem[len(base):]
        if sufijo == "" or (sufijo.startswith("_s") and sufijo[2:].isdigit()):
            accs.append(json.loads(ruta.read_text(encoding="utf-8"))["test_accuracy"])
    if accs:
        a = np.array(accs)
        bloque.append(f"{etiqueta:36s}: {len(a)} corridas, {a.mean():.1%} ± {a.std(ddof=1) if len(a) > 1 else 0:.1%}"
                      f"   ({', '.join(f'{v:.1%}' for v in a)})")
print("\n".join(bloque))
informe += [""] + bloque

(DIR_RESULTADOS / "paso5_evaluacion.txt").write_text("\n".join(informe), encoding="utf-8")
print("\nListo. Revisa resultados/")
