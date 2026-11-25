"""
exportar_onnx.py - Convierte el modelo final a ONNX para correrlo en el navegador.

Genera dos archivos en web/public/modelos/:
  cnn_rapido.onnx   1 red (semilla 42), sin espejo                 = clasificador rapido=True
  cnn_preciso.onnx  3 redes x (normal + espejo), promedio de las 6   = clasificador por defecto

Para no perder precisión:
  - float32, sin cuantizar (los mismos pesos de los .keras).
  - El redimensionado a 224x224 (bilineal, sin mantener proporción) va DENTRO del grafo,
    igual que tf.image.resize en clasificador.py. La web pasa el frame RGB a su tamaño real.
  - El ensemble y el espejo (TTA) también van dentro: la web hace una sola llamada.
  - Los .keras originales no se tocan; estos ONNX se pueden regenerar cuando haga falta.

Entrada del ONNX:  "imagen"        float32 [1, alto, ancho, 3], RGB 0-255 sin normalizar
Salida del ONNX:   "probabilidades" float32 [1, 5] en el orden de modelos/modelo_final.json

Uso:  .venv\\Scripts\\python.exe src\\exportar_onnx.py
Verificación:  .venv\\Scripts\\python.exe src\\verificar_onnx.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["KERAS_BACKEND"] = "tensorflow"   # los .keras se entrenaron con torch, pero el formato es el mismo

import tensorflow as tf
import tf2onnx

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parent.parent
DIR_MODELOS = RAIZ / "modelos"
DIR_SALIDA = RAIZ / "web" / "public" / "modelos"
OPSET = 17


def construir(redes, img_size, tta):
    """Modelo de exportación: imagen de cualquier tamaño -> 5 probabilidades."""
    entrada = tf.keras.Input(shape=(None, None, 3), dtype="float32", name="imagen")
    x = tf.keras.layers.Resizing(img_size, img_size, interpolation="bilinear", name="redimensionar")(entrada)
    vistas = [x, tf.keras.layers.Lambda(lambda t: tf.reverse(t, axis=[2]), name="espejo")(x)] if tta else [x]
    salidas = [red(v, training=False) for red in redes for v in vistas]
    y = salidas[0] if len(salidas) == 1 else tf.keras.layers.Average(name="promedio")(salidas)
    y = tf.keras.layers.Identity(name="probabilidades")(y)
    return tf.keras.Model(entrada, y)


def exportar(modelo, ruta):
    firma = (tf.TensorSpec([1, None, None, 3], tf.float32, name="imagen"),)
    funcion = tf.function(lambda imagen: {"probabilidades": modelo(imagen, training=False)}, input_signature=firma)
    tf2onnx.convert.from_function(funcion, input_signature=firma, opset=OPSET, output_path=str(ruta))
    print(f"  {ruta.relative_to(RAIZ)}  ({ruta.stat().st_size / 1e6:.1f} MB)")


def main():
    meta = json.loads((DIR_MODELOS / "modelo_final.json").read_text(encoding="utf-8"))
    redes = [tf.keras.models.load_model(DIR_MODELOS / f, compile=False) for f in meta["ensemble"]]
    for i, red in enumerate(redes):
        red.name = f"red_{i}"          # nombres únicos dentro del grafo
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    print("Exportando a ONNX (float32, opset %d):" % OPSET)
    exportar(construir(redes[:1], meta["img_size"], tta=False), DIR_SALIDA / "cnn_rapido.onnx")
    exportar(construir(redes, meta["img_size"], tta=bool(meta.get("tta_volteo"))), DIR_SALIDA / "cnn_preciso.onnx")

    (DIR_SALIDA / "modelo.json").write_text(json.dumps({
        "clases": meta["clases"],
        "nombres": {"bovino": "Bovino (vaca/toro)", "caballo": "Caballo", "fondo": "Sin animal (fondo)",
                    "gato": "Gato", "perro": "Perro"},
        "entrada": "imagen: float32 [1, alto, ancho, 3], RGB 0-255 (el modelo redimensiona y normaliza)",
        "salida": "probabilidades: float32 [1, 5]",
        "archivos": {"rapido": "cnn_rapido.onnx", "preciso": "cnn_preciso.onnx"},
        "test_accuracy": {"preciso": meta["test_accuracy"]},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  web/public/modelos/modelo.json")


if __name__ == "__main__":
    main()
