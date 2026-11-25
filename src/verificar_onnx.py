"""
verificar_onnx.py - Prueba de paridad: el ONNX debe dar lo MISMO que el modelo original.

Compara, foto por foto, ClasificadorAnimales (los .keras, en Python) contra los ONNX
exportados por exportar_onnx.py, en modo rápido y preciso.

Criterio: misma clase en todas las fotos y diferencia máxima < 0.0001 en cada probabilidad.

También guarda web/public/modelos/esperado.json con las probabilidades de Python, para que
la página web/public/prueba_onnx.html haga la misma comparación dentro del navegador.

Uso:
  .venv\\Scripts\\python.exe src\\verificar_onnx.py                     (las fotos de fotos_prueba/)
  .venv\\Scripts\\python.exe src\\verificar_onnx.py data\\dataset_cnn_maximo\\val   (las 500 del test)
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import onnxruntime as ort
import tensorflow as tf
from PIL import Image

# Los .keras guardan la config del optimizador de PyTorch (se entrenaron con ese backend).
# Para predecir no hace falta: se cargan sin compilar, sin tocar clasificador.py.
_cargar_original = tf.keras.models.load_model
tf.keras.models.load_model = lambda *a, **k: _cargar_original(*a, **{"compile": False, **k})

from clasificador import RAIZ, ClasificadorAnimales  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIR_ONNX = RAIZ / "web" / "public" / "modelos"
TOLERANCIA = 1e-4
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def listar(carpeta):
    """Fotos de la carpeta; si tiene subcarpetas por clase, devuelve (ruta, clase_real)."""
    carpeta = Path(carpeta)
    subdirs = sorted(d for d in carpeta.iterdir() if d.is_dir())
    if subdirs:
        return [(p, d.name) for d in subdirs for p in sorted(d.iterdir()) if p.suffix.lower() in EXT]
    return [(p, None) for p in sorted(carpeta.iterdir()) if p.suffix.lower() in EXT]


def main():
    carpeta = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "fotos_prueba"
    fotos = listar(carpeta)
    print(f"{len(fotos)} fotos de {carpeta}")
    todo_ok = True
    esperado = {}

    for modo, archivo in [("rapido", "cnn_rapido.onnx"), ("preciso", "cnn_preciso.onnx")]:
        clf = ClasificadorAnimales(rapido=(modo == "rapido"))
        sesion = ort.InferenceSession(str(DIR_ONNX / archivo), providers=["CPUExecutionProvider"])
        dif_max, distintas, aciertos_k, aciertos_o, con_etiqueta = 0.0, 0, 0, 0, 0

        for ruta, real in fotos:
            with Image.open(ruta) as im:
                rgb = np.array(im.convert("RGB"))
            p_keras = np.array([clf.predecir(rgb)["probabilidades"][c] for c in clf.clases])
            p_onnx = sesion.run(None, {"imagen": rgb[None].astype(np.float32)})[0][0]
            dif_max = max(dif_max, float(np.abs(p_keras - p_onnx).max()))
            distintas += int(p_keras.argmax() != p_onnx.argmax())
            if real is not None:
                con_etiqueta += 1
                aciertos_k += int(clf.clases[p_keras.argmax()] == real)
                aciertos_o += int(clf.clases[p_onnx.argmax()] == real)
            if real is None:
                esperado.setdefault(ruta.name, {})[modo] = [float(x) for x in p_keras]

        ok = dif_max < TOLERANCIA and distintas == 0
        todo_ok &= ok
        print(f"\n[{modo}] {archivo}")
        print(f"  diferencia máxima en una probabilidad: {dif_max:.2e}  (tolerancia {TOLERANCIA:.0e})")
        print(f"  fotos con distinta clase: {distintas}")
        if con_etiqueta:
            print(f"  accuracy Keras: {aciertos_k}/{con_etiqueta} = {aciertos_k / con_etiqueta:.1%}")
            print(f"  accuracy ONNX : {aciertos_o}/{con_etiqueta} = {aciertos_o / con_etiqueta:.1%}")
        print("  -> OK, paridad comprobada" if ok else "  -> FALLA")

    if esperado:
        (DIR_ONNX / "esperado.json").write_text(json.dumps({
            "clases": ClasificadorAnimales(rapido=True).clases, "tolerancia_navegador": 1e-3,
            "fotos": esperado}, indent=1), encoding="utf-8")
        print(f"\nGuardado {(DIR_ONNX / 'esperado.json').relative_to(RAIZ)} para prueba_onnx.html")
    sys.exit(0 if todo_ok else 1)


if __name__ == "__main__":
    main()
