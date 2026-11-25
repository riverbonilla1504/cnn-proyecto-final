"""
PASO 4 - Ensemble de 3 semillas + TTA (test-time augmentation) y elección del modelo final.

  - ENSEMBLE: las 3 CNN del paso 3 (misma arquitectura y datos, distinta semilla) se
    equivocan en fotos algo distintas. Promediando sus probabilidades, los errores de una
    los corrigen las otras. Con pocos datos suele ganar 1-3 puntos (Brigato & Iocchi 2021).
  - TTA: cada foto se evalúa normal y en espejo (volteo horizontal) y se promedian las dos
    predicciones. Un perro sigue siendo perro en espejo; cuesta solo una pasada más.

Se comparan 4 opciones con la VALIDACIÓN (15 % del dataset máximo, ~4 000 fotos) y se
elige la mejor SIN mirar el test. Después se reporta el test de la opción elegida
(y, como información, el de todas).

Salidas:
  resultados/paso4_ensemble_tta.txt
  modelos/modelo_final.json  (+ modelo_final.keras si la opción elegida es un solo modelo)

Uso:  .venv\\Scripts\\python.exe src\\paso4_ensemble_tta.py
"""
import json
import os
import shutil
import time

os.environ["DATASET"] = "dataset_cnn_maximo"      # su carpeta val = el mismo test de 500 fotos

import keras
import numpy as np

from comun import CLASES, DIR_MODELOS, DIR_RESULTADOS, cargar_datasets

SEMILLAS = ["cnn_propia_v2_receta", "cnn_propia_v2_receta_s1", "cnn_propia_v2_receta_s2"]
IMG = 224

_, val_ds, test_ds = cargar_datasets(IMG, 64, one_hot=False)
x_val = np.concatenate([x.numpy() for x, _ in val_ds]); y_val = np.concatenate([y.numpy() for _, y in val_ds])
x_test = np.concatenate([x.numpy() for x, _ in test_ds]); y_test = np.concatenate([y.numpy() for _, y in test_ds])
print(f"Validación: {len(y_val)} fotos | Test: {len(y_test)} fotos")

probs = {}   # (modelo, conjunto, volteo) -> probabilidades
for nombre in SEMILLAS:
    m = keras.models.load_model(DIR_MODELOS / f"{nombre}.keras")
    for conj, x in (("val", x_val), ("test", x_test)):
        probs[(nombre, conj, False)] = m.predict(x, batch_size=64, verbose=0)
        probs[(nombre, conj, True)] = m.predict(x[:, :, ::-1], batch_size=64, verbose=0)


def acc(p, y):
    return float((p.argmax(1) == y).mean())


def prob_opcion(modelos, tta, conj):
    vistas = [False, True] if tta else [False]
    return np.mean([probs[(m, conj, v)] for m in modelos for v in vistas], axis=0)


opciones = {}
for m in SEMILLAS:
    opciones[f"{m} solo"] = ([m], False)
    opciones[f"{m} + TTA"] = ([m], True)
opciones["Ensemble 3 semillas"] = (SEMILLAS, False)
opciones["Ensemble 3 semillas + TTA"] = (SEMILLAS, True)

filas = []
for nombre, (mods, tta) in opciones.items():
    filas.append((nombre, acc(prob_opcion(mods, tta, "val"), y_val),
                  acc(prob_opcion(mods, tta, "test"), y_test), mods, tta))
elegida = max(filas, key=lambda f: f[1])          # SOLO por validación

lineas = ["Opción                                   |   Val    |  Test", "-" * 62]
lineas += [f"{n:40s} | {v:7.2%} | {t:7.2%}{'   <- ELEGIDA (mejor validación)' if n == elegida[0] else ''}"
           for n, v, t, _, _ in filas]
solos = [t for n, _, t, _, _ in filas if n.endswith(" solo")]
lineas += ["", f"Semillas individuales en test: {np.mean(solos):.2%} ± {np.std(solos, ddof=1):.2%}"]

# Velocidad en CPU de la opción elegida (lo que importa para la cámara) -------------------
modelos_cpu = [keras.models.load_model(DIR_MODELOS / f"{m}.keras") for m in elegida[3]]
lote = x_test[:1]
for m in modelos_cpu:
    m(lote, training=False)                         # calentamiento
t0 = time.time()
for _ in range(20):
    for m in modelos_cpu:
        m(lote, training=False)
        if elegida[4]:
            m(lote[:, :, ::-1], training=False)
ms = (time.time() - t0) / 20 * 1000
lineas.append(f"Tiempo por imagen (CPU, opción elegida): {ms:.0f} ms  (~{1000 / ms:.0f} imágenes/s)")
texto = "\n".join(lineas)
print(texto)
(DIR_RESULTADOS / "paso4_ensemble_tta.txt").write_text(texto, encoding="utf-8")

# Escribir el modelo final ----------------------------------------------------------------
meta_base = json.loads((DIR_MODELOS / f"{SEMILLAS[0]}.json").read_text(encoding="utf-8"))
meta = {"clases": CLASES, "img_size": IMG, "entrada": meta_base["entrada"],
        "tipo": f"{elegida[0]} (CNN propia desde cero, sin modelos preentrenados)",
        "val_accuracy": round(elegida[1], 4), "test_accuracy": round(elegida[2], 4)}
for viejo in ("modelo_final.keras",):
    (DIR_MODELOS / viejo).unlink(missing_ok=True)
if len(elegida[3]) > 1 or elegida[4]:
    meta.update({"ensemble": [f"{m}.keras" for m in elegida[3]], "tta_volteo": elegida[4]})
else:
    shutil.copy2(DIR_MODELOS / f"{elegida[3][0]}.keras", DIR_MODELOS / "modelo_final.keras")
(DIR_MODELOS / "modelo_final.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nModelo final: {elegida[0]}  ->  modelos/modelo_final.json")
