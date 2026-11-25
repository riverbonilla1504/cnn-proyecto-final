"""
EXPERIMENTO E1 - Explorar el dataset original antes de entrenar.

Preguntas que respondemos aquí:
  1. ¿Cuántas imágenes hay por clase? (¿está balanceado?)
  2. ¿De qué tamaño son las imágenes? (para elegir IMG_SIZE)
  3. ¿Cómo se ven? (detectar basura, etiquetas malas, etc.)

Salidas:
  historial/resultados/exp1_conteo_clases.png
  historial/resultados/exp1_ejemplos.png
  historial/resultados/exp1_resumen.txt

Uso:  .venv\\Scripts\\python.exe historial\\src\\exp1_explorar_datos.py
"""
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from comun import CLASES, DIR_RESULTADOS, DIR_TEST, DIR_TRAIN

random.seed(42)
lineas = []


def log(txt=""):
    print(txt)
    lineas.append(txt)


# 1. Conteo de imágenes por clase ------------------------------------------------
conteo = {}
for nombre, carpeta in [("train", DIR_TRAIN), ("test (val original)", DIR_TEST)]:
    conteo[nombre] = {c: len(list((carpeta / c).glob("*.jpg"))) for c in CLASES}
    log(f"{nombre}: {conteo[nombre]}  total={sum(conteo[nombre].values())}")

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(CLASES))
ax.bar(x - 0.2, list(conteo["train"].values()), 0.4, label="train")
ax.bar(x + 0.2, list(conteo["test (val original)"].values()), 0.4, label="test")
ax.set_xticks(x, CLASES); ax.set_ylabel("Nº de imágenes"); ax.legend()
ax.set_title("Imágenes por clase")
fig.tight_layout(); fig.savefig(DIR_RESULTADOS / "exp1_conteo_clases.png", dpi=110); plt.close(fig)

# 2. Tamaños de las imágenes ------------------------------------------------------
anchos, altos = [], []
for c in CLASES:
    for f in (DIR_TRAIN / c).glob("*.jpg"):
        with Image.open(f) as im:
            anchos.append(im.width); altos.append(im.height)
anchos, altos = np.array(anchos), np.array(altos)
log()
log(f"Ancho  -> min {anchos.min()}, mediana {int(np.median(anchos))}, max {anchos.max()}")
log(f"Alto   -> min {altos.min()}, mediana {int(np.median(altos))}, max {altos.max()}")
log(f"Imágenes con algún lado < 64 px: {int(((anchos < 64) | (altos < 64)).sum())}")
log("Todas se redimensionarán a un tamaño fijo (IMG_SIZE) porque la CNN necesita entradas iguales.")

# 3. Mosaico de ejemplos ----------------------------------------------------------
fig, axes = plt.subplots(len(CLASES), 6, figsize=(12, 10))
for i, c in enumerate(CLASES):
    archivos = random.sample(list((DIR_TRAIN / c).glob("*.jpg")), 6)
    for j, f in enumerate(archivos):
        axes[i, j].imshow(Image.open(f).convert("RGB"))
        axes[i, j].axis("off")
        if j == 0:
            axes[i, j].set_title(c, loc="left", fontsize=12, fontweight="bold")
fig.suptitle("Ejemplos aleatorios por clase (recortes de Open Images)")
fig.tight_layout(); fig.savefig(DIR_RESULTADOS / "exp1_ejemplos.png", dpi=90); plt.close(fig)

(DIR_RESULTADOS / "exp1_resumen.txt").write_text("\n".join(lineas), encoding="utf-8")
print("\nListo. Revisa la carpeta resultados/")
