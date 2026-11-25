"""
EXPERIMENTO E4 - Curva de aprendizaje: ¿cuánto mejora la CNN al darle MÁS fotos?

Se entrena la MISMA CNN v2 (experimento E3) con subconjuntos crecientes del dataset ampliado
(paso 1 del pipeline final) y se mide siempre con el MISMO test de 500 fotos:
    500 por clase (= dataset original) -> 1 000 -> 1 500 -> todas (~2 300)
Los subconjuntos son ANIDADOS (el de 1 000 contiene al de 500, etc.), así lo único que
cambia entre puntos de la curva es la cantidad de datos.

Si la curva sigue subiendo al final -> conseguir más datos todavía ayudaría.
Si se aplana -> el límite ya no son los datos sino el modelo.

Uso:
  1) Crear los subconjuntos (hard links: no ocupan espacio extra en disco):
       .venv\\Scripts\\python.exe historial\\src\\exp4_curva_aprendizaje.py preparar
  2) Entrenar cada uno con el experimento E3, por ejemplo:
       $env:DATASET = "dataset_cnn_n1000"; .venv\\Scripts\\python.exe historial\\src\\exp3_entrenar_cnn_v2.py
  3) Graficar:
       .venv\\Scripts\\python.exe historial\\src\\exp4_curva_aprendizaje.py graficar
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

from comun import CLASES, DIR_MODELOS, DIR_RESULTADOS, RAIZ, RAIZ_PROYECTO

AMPLIADO = RAIZ_PROYECTO / "data" / "dataset_cnn_ampliado"
TAMANOS = [1000, 1500]           # 500 = dataset original; "todas" = dataset ampliado


def preparar():
    man = pd.read_csv(AMPLIADO / "manifest_train.csv")
    for n in TAMANOS:
        destino = RAIZ / "data" / f"dataset_cnn_n{n}"
        shutil.rmtree(destino, ignore_errors=True)
        for r in man[man.orden < n].itertuples():
            dst = destino / r.archivo
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(AMPLIADO / r.archivo, dst)
        shutil.copytree(AMPLIADO / "val", destino / "val", copy_function=os.link)
        print(destino.name, {c: len(list((destino / "train" / c).glob("*.jpg"))) for c in CLASES})


def graficar():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    puntos = []   # (fotos de entrenamiento por clase, lista de accuracies)
    grupos = {"cnn_propia_v2": None, **{f"cnn_propia_v2_n{n}": n for n in TAMANOS},
              "cnn_propia_v2_ampliado": None}
    for base, n in grupos.items():
        accs = []
        for ruta in sorted(DIR_MODELOS.glob(f"{base}*.json")):
            suf = ruta.stem[len(base):]
            if suf == "" or (suf.startswith("_s") and suf[2:].isdigit()):
                accs.append(json.loads(ruta.read_text(encoding="utf-8"))["test_accuracy"])
        if not accs:
            continue
        if base == "cnn_propia_v2":
            n = 500
        elif base.endswith("ampliado"):
            m = pd.read_csv(AMPLIADO / "manifest_train.csv")
            n = int(m.groupby("clase").size().mean())
        puntos.append((n, accs))
    puntos.sort()

    lineas = ["Curva de aprendizaje - CNN desde cero v2 (test fijo de 500 fotos)",
              "fotos/clase (archivo)  | corridas | test promedio ± desv."]
    for n, a in puntos:
        a = np.array(a)
        lineas.append(f"{n:>11} | {len(a):^8} | {a.mean():.1%} ± {a.std(ddof=1) if len(a) > 1 else 0:.1%}"
                      f"   ({', '.join(f'{x:.1%}' for x in a)})")
    texto = "\n".join(lineas)
    print(texto)
    (DIR_RESULTADOS / "exp4_curva_aprendizaje.txt").write_text(texto, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [n for n, _ in puntos]
    medias = [np.mean(a) * 100 for _, a in puntos]
    for n, a in puntos:
        ax.scatter([n] * len(a), np.array(a) * 100, color="gray", alpha=.5, s=18)
    ax.plot(xs, medias, "o-", lw=2, label="CNN desde cero v2 (promedio)")
    ax.set_xlabel("Fotos de entrenamiento por clase (≈85 % entrena, 15 % valida)")
    ax.set_ylabel("Accuracy en test (%)")
    ax.set_title("Curva de aprendizaje: ¿ayuda tener más datos?")
    ax.grid(alpha=.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(DIR_RESULTADOS / "exp4_curva_aprendizaje.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    {"preparar": preparar, "graficar": graficar}[sys.argv[1] if len(sys.argv) > 1 else "graficar"]()
