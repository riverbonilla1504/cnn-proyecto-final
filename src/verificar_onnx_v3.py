"""
verificar_onnx_v3.py - Prueba de paridad: el ONNX de la v3 debe dar lo MISMO que Keras.

Para cada foto de fotos_prueba/ y algunos frames de video:
  Keras  (los .keras del ensemble, mismo redimensionado, misma calibración por animal)
  ONNX   (web/public/modelos/cnn_preciso.onnx y cnn_rapido.onnx, con onnxruntime)
y compara las 5 probabilidades. Criterio: diferencia máxima < 0.001 y la misma clase.
También mide cuánto tarda cada ONNX en CPU (el navegador con WASM suele ser 2-3 veces más lento).

Uso:  .venv-onnx\\Scripts\\python.exe src\\verificar_onnx_v3.py
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["KERAS_BACKEND"] = "tensorflow"

import cv2
import keras
import numpy as np
import onnxruntime as ort
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capas  # noqa: F401,E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parent.parent
ANIMALES = ["bovino", "caballo", "gato", "perro"]
CLASES = ["bovino", "caballo", "fondo", "gato", "perro"]
meta = json.loads((RAIZ / "modelos" / "modelo_final_v3.json").read_text(encoding="utf-8"))
web = json.loads((RAIZ / "web" / "public" / "modelos" / "modelo.json").read_text(encoding="utf-8"))
redes = [keras.models.load_model(RAIZ / "modelos" / f, compile=False) for f in meta["ensemble"]]
U = np.array([meta["umbrales_por_animal"][a] for a in ANIMALES])
T = web["umbral_recomendado"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def keras_5(rgb, lado, nets):
    h, w = rgb.shape[:2]
    e = min(1.0, lado / max(h, w))
    x = tf.image.resize(rgb[None].astype("float32"), (round(h * e), round(w * e)), method="bilinear")
    p = np.mean([n(x, training=False).numpy()[0] for n in nets], axis=0)
    p = 1 / (1 + np.exp(-(logit(p) - logit(U) + logit(T))))
    return np.concatenate([p[:2], [1 - p.max()], p[2:]])


imagenes = [(f.name, cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)) for f in sorted((RAIZ / "fotos_prueba").glob("*.jpg"))]
for v in sorted((RAIZ / "web" / "public" / "demo").glob("*.mp4")):
    c = cv2.VideoCapture(str(v)); c.set(1, int(c.get(7) // 2)); ok, fr = c.read()
    if ok:
        imagenes.append((f"{v.name} (frame)", cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))

peor = 0.0
for archivo, lado, nets in (("cnn_preciso.onnx", meta["lado_preciso"], redes),
                            ("cnn_rapido.onnx", meta["lado_rapido"], redes[1:2] or redes[:1])):
    s = ort.InferenceSession(str(RAIZ / "web" / "public" / "modelos" / archivo), providers=["CPUExecutionProvider"])
    print(f"\n== {archivo} (lado {lado})")
    tiempos = []
    for nombre, rgb in imagenes:
        t0 = time.perf_counter()
        o = s.run(None, {"imagen": rgb[None].astype("float32")})
        tiempos.append(time.perf_counter() - t0)
        salidas = dict(zip([x.name for x in s.get_outputs()], o))
        p_onnx = salidas["probabilidades"][0]
        p_k = keras_5(rgb, lado, nets)
        d = float(np.abs(p_onnx - p_k).max())
        peor = max(peor, d)
        igual = CLASES[int(p_onnx.argmax())] == CLASES[int(p_k.argmax())]
        print(f"  {nombre:32s} onnx {CLASES[int(p_onnx.argmax())]:7s} {p_onnx.max():.3f} | keras {CLASES[int(p_k.argmax())]:7s} "
              f"{p_k.max():.3f} | dif máx {d:.5f} {'OK' if igual and d < 1e-3 else 'REVISAR'} | mapa {salidas['mapa'].shape}")
    print(f"  tiempo ONNX en CPU: mediana {np.median(tiempos[1:]) * 1000:.0f} ms por imagen")
print(f"\nDiferencia máxima Keras vs ONNX: {peor:.6f} -> {'PASA' if peor < 1e-3 else 'NO PASA'}")
