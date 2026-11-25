"""
PASO 8 - Comparar la CNN v3 con el modelo anterior (v2, modo rápido) donde importa: el VIDEO.

Tres pruebas, ninguna usada para entrenar:
  A. Test de Open Images (las mismas 500 fotos de siempre, el animal llena la foto).
     Sirve para comprobar que la v3 no empeoró en lo que la v2 ya hacía bien.
  B. Test de ESCENAS: fotos completas de cámaras trampa (Caltech, WCS, NACTI) de
     ubicaciones que NUNCA se usaron para entrenar, y fotos de COCO val2017.
       - v2: como la web actual, el frame entero aplastado a 224x224.
       - v3: el frame completo a 640 px de lado mayor, sin aplastar.
     Métricas: ¿detecta que hay animal? ¿acierta la especie? ¿falsas alarmas en fondo?
  C. Videos de prueba (videos_prueba/): % de frames donde reconoce la vaca.

El umbral de la v3 se elige con la VALIDACIÓN (escenas de otras cámaras), nunca con el test.

Uso:   .venv\\Scripts\\python.exe src\\paso8_evaluar_v3.py [--modelo cnn_v3]
Salida: resultados/paso8_evaluacion_v3.txt
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import sys
from pathlib import Path

import cv2
import keras
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import capas  # noqa: F401  (registra PoolingLSE para poder cargar la v3)

RAIZ = Path(__file__).resolve().parent.parent
ANIMALES = ["bovino", "caballo", "gato", "perro"]
CLASES_V2 = ["bovino", "caballo", "fondo", "gato", "perro"]
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--modelos", default="cnn_v3", help="uno o varios separados por coma (ensemble = promedio)")
ap.add_argument("--lado", type=int, default=640, help="lado mayor del frame para la v3")
ap.add_argument("--guardar", action="store_true", help="guardar modelos/modelo_final_v3.json con umbrales")
args = ap.parse_args()
NOMBRES_MOD = args.modelos.split(",")
NOMBRE_INF = "+".join(NOMBRES_MOD)
redes_v3 = [keras.models.load_model(RAIZ / "modelos" / f"{m}.keras") for m in NOMBRES_MOD]
v2 = keras.models.load_model(RAIZ / "modelos" / "cnn_propia_v2_receta.keras")
informe = []


def decir(t=""):
    print(t)
    informe.append(t)


def leer_rgb(ruta):
    return cv2.cvtColor(cv2.imread(str(ruta)), cv2.COLOR_BGR2RGB)


def a_lado(rgb, lado):
    h, w = rgb.shape[:2]
    f = lado / max(h, w)
    return cv2.resize(rgb, (max(32, round(w * f)), max(32, round(h * f))), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)


def pred_v3(rgb, lado):
    x = a_lado(rgb, lado).astype("float32")[None]
    return np.mean([keras.ops.convert_to_numpy(r(x, training=False))[0] for r in redes_v3], axis=0)   # 4 prob.


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def pred_v2(rgb):
    x = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype("float32")[None]
    p = keras.ops.convert_to_numpy(v2(x, training=False))[0]
    return np.array([p[CLASES_V2.index(a)] for a in ANIMALES]), p            # 4 animales + softmax completo


def etiquetas_v2(p5, umbral=0.6):
    c = CLASES_V2[int(np.argmax(p5))]
    return set() if c == "fondo" or p5.max() < umbral else {c}


def etiquetas_v3(p4, umbral):
    """umbral: un número (igual para los 4) o un vector con un umbral por animal."""
    u = np.broadcast_to(np.asarray(umbral, dtype=float), (len(ANIMALES),))
    return {a for a, p, uu in zip(ANIMALES, p4, u) if p >= uu}


# ------------------------------------------------------------------ escenas (val y test)
man = pd.read_csv(RAIZ / "data" / "dataset_v3" / "manifest.csv")


def predecir_split(sp):
    df = man[man.split == sp].reset_index(drop=True)
    P3, P2, P2c = [], [], []
    for a in df.archivo:
        rgb = leer_rgb(RAIZ / "data" / "dataset_v3" / a)
        P3.append(pred_v3(rgb, args.lado))
        p4, p5 = pred_v2(rgb)
        P2.append(p4); P2c.append(p5)
    return df, np.array(P3), np.array(P2), np.array(P2c)


def resumen(df, sets_pred, nombre):
    """Métricas de escena: hay/no hay animal, especie correcta y falsas alarmas."""
    Y = [set(a for a in ANIMALES if f[a]) for _, f in df.iterrows()]
    con = [i for i, y in enumerate(Y) if y]
    sin = [i for i, y in enumerate(Y) if not y]
    detecta = np.mean([bool(sets_pred[i]) for i in con]) if con else float("nan")
    especie = np.mean([bool(sets_pred[i] & Y[i]) for i in con]) if con else float("nan")
    exacta = np.mean([sets_pred[i] == Y[i] for i in range(len(Y))])
    falsas = np.mean([bool(sets_pred[i]) for i in sin]) if sin else float("nan")
    return f"{nombre:26s} detecta animal {detecta:6.1%} | especie correcta {especie:6.1%} | " \
           f"escena exacta {exacta:6.1%} | falsas alarmas en fondo {falsas:6.1%}"


print("Prediciendo validación (para elegir el umbral)...")
dfv, P3v, _, _ = predecir_split("val")
Yv = dfv[ANIMALES].values
mejor_u, mejor_e = 0.5, -1
for u in np.arange(0.2, 0.91, 0.05):
    e = np.mean([(P3v[i] >= u).astype(int).tolist() == Yv[i].tolist() for i in range(len(Yv))])
    if e > mejor_e:
        mejor_u, mejor_e = round(float(u), 2), e
decir(f"Modelo(s) v3: {NOMBRE_INF} | frame a {args.lado} px")
decir(f"Umbral ÚNICO elegido con validación ({len(dfv)} escenas): {mejor_u}  (escena exacta en val {mejor_e:.1%})")
# Umbral POR ANIMAL (también con validación): para cada clase, el que da mejor F1.
# Corrige que la red diga "caballo" de más: a esa clase se le exige más.
umbrales = []
for j, a in enumerate(ANIMALES):
    mejor = (0.5, -1)
    for u in np.arange(0.2, 0.96, 0.025):
        tp = np.sum((P3v[:, j] >= u) & (Yv[:, j] == 1)); fp = np.sum((P3v[:, j] >= u) & (Yv[:, j] == 0))
        fn = np.sum((P3v[:, j] < u) & (Yv[:, j] == 1))
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > mejor[1]:
            mejor = (round(float(u), 3), f1)
    umbrales.append(mejor[0])
umbrales = np.array(umbrales)
e_pc = np.mean([(P3v[i] >= umbrales).astype(int).tolist() == Yv[i].tolist() for i in range(len(Yv))])
decir(f"Umbral POR ANIMAL (validación, mejor F1): {dict(zip(ANIMALES, umbrales.tolist()))}  (escena exacta en val {e_pc:.1%})")

print("Prediciendo test...")
dft, P3t, P2t, P2ct = predecir_split("test")
decir("\n===== B. Test de ESCENAS (fotos completas de cámaras nunca vistas + COCO) =====")
for fuente in ["cct", "wcs", "nacti", "coco", None]:
    idx = dft.index if fuente is None else dft.index[dft.fuente == fuente]
    sub = dft.loc[idx]
    nombre = "TODAS" if fuente is None else fuente
    decir(f"-- {nombre} ({len(sub)} fotos)")
    decir("  " + resumen(sub, [etiquetas_v2(P2ct[i]) for i in idx], "v2 (frame aplastado 224)"))
    decir("  " + resumen(sub, [etiquetas_v3(P3t[i], mejor_u) for i in idx], f"v3 umbral único {mejor_u}"))
    decir("  " + resumen(sub, [etiquetas_v3(P3t[i], umbrales) for i in idx], "v3 umbral por animal"))
decir("\nAP por clase en test (área bajo precisión-recall; 1.0 = perfecto):")
for j, a in enumerate(ANIMALES):
    y = dft[a].values
    if y.sum():
        decir(f"  {a:8s} v2 {average_precision_score(y, P2t[:, j]):.3f}   v3 {average_precision_score(y, P3t[:, j]):.3f}   ({int(y.sum())} fotos)")

# ------------------------------------------------------------------ Open Images (500 recortes)
decir("\n===== A. Test de Open Images (500 fotos, el animal llena la foto) =====")
dir_oi = RAIZ / "data" / "dataset_cnn_maximo" / "val"
ok2 = ok3 = ok3c = n = 0
for c in CLASES_V2:
    for f in sorted((dir_oi / c).glob("*.jpg")):
        rgb = leer_rgb(f)
        _, p5 = pred_v2(rgb)
        p4 = pred_v3(cv2.resize(rgb, (320, 320)), 320)
        pred3 = ANIMALES[int(np.argmax(p4))] if p4.max() >= mejor_u else "fondo"
        s = logit(p4) - logit(umbrales)                   # >0 = supera el umbral de su clase
        pred3c = ANIMALES[int(np.argmax(s))] if s.max() >= 0 else "fondo"
        ok2 += CLASES_V2[int(np.argmax(p5))] == c
        ok3 += pred3 == c
        ok3c += pred3c == c
        n += 1
decir(f"  v2: {ok2 / n:.1%}   v3 umbral único: {ok3 / n:.1%}   v3 umbral por animal: {ok3c / n:.1%}   ({n} fotos)")

# ------------------------------------------------------------------ videos
OBJETIVO = {"vacas_carretera": "bovino", "vaca_desierto": "bovino", "perros_calle": "perro", "caballos_pradera": "caballo"}
decir("\n===== C. Videos de prueba (% de frames con la especie correcta / con alguna especie equivocada) =====")
for video in sorted((RAIZ / "videos_prueba").glob("*.mp4")):
    obj = OBJETIVO.get(video.stem)
    if obj is None:
        continue
    cap = cv2.VideoCapture(str(video))
    cuenta = {"v2": [0, 0], "v3": [0, 0]}
    total = vistos = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        total += 1
        if total % 3:
            continue
        vistos += 1
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        for nombre, et in (("v2", etiquetas_v2(pred_v2(rgb)[1])), ("v3", etiquetas_v3(pred_v3(rgb, args.lado), umbrales))):
            cuenta[nombre][0] += obj in et
            cuenta[nombre][1] += bool(et - {obj})
    m = max(1, vistos)
    decir(f"  {video.name:22s} ({obj:7s}) v2 correcta {cuenta['v2'][0] / m:6.1%} equivocada {cuenta['v2'][1] / m:6.1%} | "
          f"v3 correcta {cuenta['v3'][0] / m:6.1%} equivocada {cuenta['v3'][1] / m:6.1%}  ({m} frames)")

if args.guardar:
    import json
    (RAIZ / "modelos" / "modelo_final_v3.json").write_text(json.dumps({
        "ensemble": [f"{m}.keras" for m in NOMBRES_MOD], "clases": ANIMALES,
        "umbral": mejor_u, "umbrales_por_animal": dict(zip(ANIMALES, umbrales.tolist())),
        "lado_rapido": 480, "lado_preciso": args.lado,
        "metricas": {"escena_exacta_val": round(float(e_pc), 4)},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    decir("\nGuardado modelos/modelo_final_v3.json")

(RAIZ / "resultados" / f"paso8_evaluacion_{NOMBRE_INF}_{args.lado}.txt").write_text("\n".join(informe), encoding="utf-8")
print("\nListo -> resultados/")
