"""
exportar_onnx_v3.py - Convierte la CNN v3 a ONNX para correrla en el navegador.

Igual que exportar_onnx.py (v2), pero para la v3:
  - La v3 es TOTALMENTE CONVOLUCIONAL: no se aplasta el frame a 224x224. Dentro del grafo
    se reduce manteniendo la proporción hasta que el lado mayor mida LADO px (640 o 480).
  - Sigue entregando "probabilidades" [1, 5] en el MISMO orden que la v2
    (bovino, caballo, fondo, gato, perro), con fondo = 1 - el animal más probable.
    Así la web de Javier (modelo.js -> interpretar) funciona sin cambios.
  - Salidas nuevas:
      "animales" [1, 4]          sigmoide por animal (bovino, caballo, gato, perro): puede haber varios.
      "mapa"     [1, h, w, 4]    probabilidad por ZONA de la imagen: dice DÓNDE está cada animal.
  - float32, sin cuantizar. Los .keras no se tocan.

Archivos que genera en web/public/modelos/:
  cnn_rapido.onnx   la mejor semilla sola, lado mayor 480 px
  cnn_preciso.onnx  promedio de las semillas (ensemble), lado mayor 640 px

Uso:  .venv\\Scripts\\python.exe src\\exportar_onnx_v3.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import tf2onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capas  # noqa: F401,E402  (registra PoolingLSE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parent.parent
DIR_MODELOS = RAIZ / "modelos"
DIR_SALIDA = RAIZ / "web" / "public" / "modelos"
OPSET = 17


def logit(p):
    p = tf.clip_by_value(p, 1e-6, 1 - 1e-6)
    return tf.math.log(p / (1 - p))


def construir(redes, lado, umbrales, umbral_web):
    """imagen RGB 0-255 de cualquier tamaño -> (probabilidades 5, animales 4, mapa h x w x 4).

    Calibración: en validación cada animal tiene su propio umbral (p. ej. "caballo" 0.75,
    "bovino" 0.575) porque la red decía caballo de más. Se mueven las probabilidades en la
    escala logit para que TODOS los animales queden en el mismo umbral de la web:
        p' = sigmoide( logit(p) - logit(umbral_animal) + logit(umbral_web) )
    Así, p' >= umbral_web  <=>  p >= umbral_animal, y la web no necesita saber nada de esto."""
    sub = [keras.Model(r.input, [r.get_layer("mapa_evidencias").output, r.output]) for r in redes]
    import math
    lg = lambda p: math.log(p / (1 - p))
    desp = [[lg(umbral_web) - lg(u) for u in umbrales]]                 # números de Python -> constantes del grafo

    def f(imagen):
        desplazamiento = tf.constant(desp, tf.float32)                   # [1, 4], creado DENTRO del grafo
        h = tf.cast(tf.shape(imagen)[1], tf.float32)
        w = tf.cast(tf.shape(imagen)[2], tf.float32)
        escala = tf.minimum(1.0, lado / tf.maximum(h, w))           # solo reduce, nunca agranda
        tam = tf.cast(tf.round(tf.stack([h, w]) * escala), tf.int32)
        x = tf.image.resize(imagen, tam, method="bilinear")
        mapas, probs = zip(*[s(x, training=False) for s in sub])
        animales = tf.add_n(list(probs)) / len(probs)                  # [1, 4] promedio del ensemble
        animales = tf.sigmoid(logit(animales) + desplazamiento)        # calibrado por animal
        mapa = tf.sigmoid(tf.add_n(list(mapas)) / len(mapas) + desplazamiento[:, None, None, :])
        fondo = 1.0 - tf.reduce_max(animales, axis=-1, keepdims=True)
        cinco = tf.concat([animales[:, :2], fondo, animales[:, 2:]], axis=-1)
        return {"probabilidades": cinco, "animales": animales, "mapa": mapa}
    return f


UMBRAL_WEB = 0.6     # valor por defecto del deslizador de la web


def exportar(redes, lado, umbrales, ruta):
    firma = (tf.TensorSpec([1, None, None, 3], tf.float32, name="imagen"),)
    funcion = tf.function(construir(redes, lado, umbrales, UMBRAL_WEB), input_signature=firma)
    tf2onnx.convert.from_function(funcion, input_signature=firma, opset=OPSET, output_path=str(ruta))
    print(f"  {ruta.relative_to(RAIZ)}  ({ruta.stat().st_size / 1e6:.1f} MB)")


def main():
    meta = json.loads((DIR_MODELOS / "modelo_final_v3.json").read_text(encoding="utf-8"))
    redes = [keras.models.load_model(DIR_MODELOS / f, compile=False) for f in meta["ensemble"]]
    for i, r in enumerate(redes):
        r.name = f"red_{i}"
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    print("Exportando la CNN v3 a ONNX (float32, opset %d):" % OPSET)
    umbrales = [meta["umbrales_por_animal"][a] for a in ["bovino", "caballo", "gato", "perro"]]
    print("Umbrales por animal (validación):", dict(zip(["bovino", "caballo", "gato", "perro"], umbrales)))
    exportar(redes[1:2] or redes[:1], meta["lado_rapido"], umbrales, DIR_SALIDA / "cnn_rapido.onnx")
    exportar(redes, meta["lado_preciso"], umbrales, DIR_SALIDA / "cnn_preciso.onnx")
    (DIR_SALIDA / "modelo.json").write_text(json.dumps({
        "version": "v3",
        "clases": ["bovino", "caballo", "fondo", "gato", "perro"],
        "animales": ["bovino", "caballo", "gato", "perro"],
        "nombres": {"bovino": "Bovino (vaca/toro)", "caballo": "Caballo", "fondo": "Sin animal (fondo)",
                    "gato": "Gato", "perro": "Perro"},
        "entrada": "imagen: float32 [1, alto, ancho, 3], RGB 0-255 de cualquier tamaño (el modelo reduce y normaliza)",
        "salidas": {"probabilidades": "[1, 5] mismo orden que la v2 (fondo = 1 - animal más probable)",
                    "animales": "[1, 4] sigmoide por animal (multi-etiqueta)",
                    "mapa": "[1, h, w, 4] probabilidad por zona (dónde está cada animal)"},
        "archivos": {"rapido": "cnn_rapido.onnx", "preciso": "cnn_preciso.onnx"},
        "lado": {"rapido": meta["lado_rapido"], "preciso": meta["lado_preciso"]},
        "umbral_recomendado": UMBRAL_WEB,
        "umbrales_por_animal_calibrados": meta["umbrales_por_animal"],
        "metricas": meta.get("metricas", {}),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  web/public/modelos/modelo.json")


if __name__ == "__main__":
    main()
