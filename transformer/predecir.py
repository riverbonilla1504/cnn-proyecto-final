"""Prueba el modelo ONNX de consultas desde la terminal."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from generar_dataset import codificar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preguntas", nargs="+", help="Escribe cada pregunta entre comillas.")
    args = parser.parse_args()

    publicado = Path(__file__).resolve().parents[1] / "web" / "public" / "chat"
    datos = json.loads((publicado / "vocab.json").read_text(encoding="utf-8"))
    entrada = np.array(
        [codificar(pregunta, datos["tokens"], datos["longitud"]) for pregunta in args.preguntas],
        dtype=np.int32,
    )
    sesion = ort.InferenceSession(str(publicado / "transformer.onnx"), providers=["CPUExecutionProvider"])
    nombres = tuple(datos["etiquetas"])
    result = sesion.run([f"prob_{nombre}" for nombre in nombres], {"tokens": entrada})
    for i, pregunta in enumerate(args.preguntas):
        prediccion = {"pregunta": pregunta}
        for nombre, probabilidades in zip(nombres, result, strict=True):
            indice = int(probabilidades[i].argmax())
            prediccion[nombre] = datos["etiquetas"][nombre][indice]
            prediccion[f"confianza_{nombre}"] = round(float(probabilidades[i, indice]), 4)
        print(json.dumps(prediccion, ensure_ascii=False))


if __name__ == "__main__":
    main()
