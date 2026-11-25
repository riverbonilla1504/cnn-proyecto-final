"""Entrena desde cero y exporta el clasificador de consultas de DetAny3D."""

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from keras import layers, ops

from generar_dataset import (
    CLASES, INTENCIONES, PERIODOS, alias_lexicos, codificar, generar, generar_refuerzo, tokens,
)

RAIZ = Path(__file__).resolve().parent
PUBLICO = RAIZ.parent / "web" / "public" / "chat"
LONGITUD = 24
CABEZAS = ("intencion", "clase", "periodo")
ETIQUETAS = (INTENCIONES, tuple(CLASES), tuple(PERIODOS))


@keras.saving.register_keras_serializable(package="DetAny3D")
class EmbeddingConPosicion(layers.Layer):
    def __init__(self, vocabulario, dimension, **kwargs):
        super().__init__(**kwargs)
        self.vocabulario = vocabulario
        self.dimension = dimension
        self.palabras = layers.Embedding(vocabulario, dimension)
        self.posiciones = layers.Embedding(LONGITUD, dimension)

    def build(self, input_shape):
        self.palabras.build(input_shape)
        self.posiciones.build((LONGITUD,))
        super().build(input_shape)

    def call(self, entradas):
        return self.palabras(entradas) + self.posiciones(ops.arange(LONGITUD))

    def get_config(self):
        return {
            **super().get_config(),
            "vocabulario": self.vocabulario,
            "dimension": self.dimension,
        }


def crear_modelo(vocabulario):
    entrada = keras.Input(shape=(LONGITUD,), dtype="int32", name="tokens")
    mascara = ops.not_equal(entrada, 0)
    mascara_atencion = ops.expand_dims(mascara, axis=1)
    mascara_lectura = ops.expand_dims(mascara, axis=-1)
    dimension = 64
    x = EmbeddingConPosicion(vocabulario, dimension)(entrada)
    x = layers.Dropout(0.06, noise_shape=(None, LONGITUD, 1))(x)
    for _ in range(3):
        atencion = layers.MultiHeadAttention(num_heads=8, key_dim=8, dropout=0.1)(
            x, x, attention_mask=mascara_atencion
        )
        x = layers.LayerNormalization(epsilon=1e-5)(layers.Add()([x, atencion]))
        densa = layers.Dense(2 * dimension, activation="gelu")(x)
        densa = layers.Dropout(0.1)(densa)
        densa = layers.Dense(dimension)(densa)
        x = layers.LayerNormalization(epsilon=1e-5)(layers.Add()([x, densa]))

    salidas = []
    for nombre, clases in zip(CABEZAS, ETIQUETAS, strict=True):
        # Cada tarea aprende dónde leer su evidencia en la misma pregunta.
        puntuacion = layers.Dense(1)(layers.Dense(32, activation="tanh")(x))
        pesos = layers.Softmax(axis=1, name=f"atencion_{nombre}")(
            puntuacion, mask=mascara_lectura
        )
        resumen = ops.sum(x * pesos, axis=1)
        representacion = layers.Dense(dimension, activation="gelu")(resumen)
        salidas.append(layers.Dense(len(clases), activation="softmax", name=nombre)(representacion))
    modelo = keras.Model(entrada, salidas, name="detany3d_consultas")
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=["sparse_categorical_crossentropy"] * len(CABEZAS),
    )
    return modelo


def vectorizar(preguntas, vocabulario):
    return np.array([codificar(texto, vocabulario, LONGITUD) for texto in preguntas], dtype=np.int32)


def preparar(ejemplos, vocabulario):
    textos, etiquetas = zip(*ejemplos, strict=True)
    codificadas = tuple(
        np.array([clases.index(etiqueta[i]) for etiqueta in etiquetas], dtype=np.int32)
        for i, clases in enumerate(ETIQUETAS)
    )
    return vectorizar(textos, vocabulario), codificadas


def leer_evaluacion(nombre):
    with (RAIZ / nombre).open(encoding="utf-8", newline="") as archivo:
        return [
            (fila["pregunta"], tuple(fila[cabeza] for cabeza in CABEZAS))
            for fila in csv.DictReader(archivo, delimiter="\t")
        ]


def evaluar(probabilidades, verdaderas):
    predicciones = np.stack([prob.argmax(axis=1) for prob in probabilidades], axis=1)
    aciertos = predicciones == np.stack(verdaderas, axis=1)
    return {
        "ejemplos": len(aciertos),
        "exacta": float(aciertos.all(axis=1).mean()),
        **{nombre: float(aciertos[:, i].mean()) for i, nombre in enumerate(CABEZAS)},
    }, predicciones


def detalle(ejemplos, predicciones):
    return [
        {
            "pregunta": pregunta,
            "real": dict(zip(CABEZAS, esperada, strict=True)),
            "predicha": {
                nombre: clases[indice]
                for nombre, clases, indice in zip(CABEZAS, ETIQUETAS, predicha, strict=True)
            },
        }
        for (pregunta, esperada), predicha in zip(ejemplos, predicciones, strict=True)
    ]


def exportar(modelo, entrada, probabilidades, destino):
    firma = [tf.TensorSpec([None, LONGITUD], tf.int32, name="tokens")]

    @tf.function(input_signature=firma)
    def inferir(ids):
        return tuple(modelo(ids, training=False))

    exportado, _ = tf2onnx.convert.from_function(inferir, input_signature=firma, opset=17)
    # ONNX 17 tiene Erf. TensorFlow usa Erfc para GELU y Erfc(x) equivale a 1 - Erf(x).
    if any(nodo.op_type == "Erfc" for nodo in exportado.graph.node):
        uno = "detany3d_erfc_uno"
        exportado.graph.initializer.append(onnx.helper.make_tensor(uno, onnx.TensorProto.FLOAT, [], [1.0]))
        nodos = []
        for nodo in exportado.graph.node:
            if nodo.op_type == "Erfc":
                erf = nodo.output[0] + "_erf"
                nodos.append(onnx.helper.make_node("Erf", nodo.input, [erf]))
                nodos.append(onnx.helper.make_node("Sub", [uno, erf], nodo.output))
            else:
                nodos.append(nodo)
        del exportado.graph.node[:]
        exportado.graph.node.extend(nodos)
    nombres = [f"prob_{nombre}" for nombre in CABEZAS]
    for salida, nombre in zip(exportado.graph.output, nombres, strict=True):
        if salida.name != nombre:
            exportado.graph.node.append(onnx.helper.make_node("Identity", [salida.name], [nombre]))
            salida.name = nombre
    onnx.checker.check_model(exportado)
    sesion = ort.InferenceSession(exportado.SerializeToString(), providers=["CPUExecutionProvider"])
    resultado = sesion.run(nombres, {"tokens": entrada})
    for keras_prob, onnx_prob in zip(probabilidades, resultado, strict=True):
        np.testing.assert_allclose(keras_prob, onnx_prob, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(keras_prob.argmax(axis=1), onnx_prob.argmax(axis=1))
    destino.write_bytes(exportado.SerializeToString())
    return max(float(np.max(np.abs(k - o))) for k, o in zip(probabilidades, resultado, strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epocas", type=int, default=12)
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--solo-exportar", action="store_true", help="Reutiliza el modelo Keras guardado.")
    modo.add_argument("--continuar", action="store_true", help="Ajusta los pesos guardados con ejemplos difíciles.")
    args = parser.parse_args()
    if args.epocas < 1:
        parser.error("--epocas debe ser positivo")
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    keras.utils.set_random_seed(42)

    pruebas = {
        "escrita": leer_evaluacion("evaluacion.tsv"),
        "generalizacion": leer_evaluacion("evaluacion_generalizacion.tsv"),
        "contrastes": leer_evaluacion("evaluacion_contrastes.tsv"),
        "contexto": leer_evaluacion("evaluacion_contexto.tsv"),
    }
    reservadas = [fila for ejemplos in pruebas.values() for fila in ejemplos]
    assert len({" ".join(tokens(texto)) for texto, _ in reservadas}) == len(reservadas)
    alias = alias_lexicos()

    def canonizar(texto):
        return " ".join(alias.get(token, token) for token in tokens(texto))

    claves_reservadas = {canonizar(texto) for texto, _ in reservadas}
    unicas = {}
    conflictos = set()
    for texto, etiqueta in generar():
        canonica = canonizar(texto)
        if canonica not in claves_reservadas and unicas.setdefault(canonica, etiqueta) != etiqueta:
            conflictos.add(canonica)
    sinteticas = [fila for fila in unicas.items() if fila[0] not in conflictos]
    corte = int(len(sinteticas) * 0.85)
    entrenamiento, validacion = sinteticas[:corte], sinteticas[corte:]
    anterior = None
    vocabulario_anterior = None
    archivo_keras = RAIZ / "modelo.keras"
    if args.solo_exportar or args.continuar:
        anterior = keras.models.load_model(archivo_keras, compile=False)
        vocabulario_anterior = json.loads((PUBLICO / "vocab.json").read_text(encoding="utf-8"))["tokens"]
    if args.continuar:
        refuerzo = {
            canonizar(texto): etiqueta for texto, etiqueta in generar_refuerzo()
            if canonizar(texto) not in claves_reservadas
        }
        disponibles = [fila for fila in entrenamiento if fila[0] not in refuerzo]
        entrenamiento = random.Random(43).sample(disponibles, min(4200, len(disponibles))) + list(refuerzo.items())
        validacion = [fila for fila in validacion if fila[0] not in refuerzo]
        random.Random(44).shuffle(entrenamiento)
    frecuencias = Counter(token for texto, _ in entrenamiento for token in tokens(texto))
    vocabulario = dict(vocabulario_anterior or {"<pad>": 0, "<unk>": 1})
    dimension_vocabulario = max(vocabulario.values()) + 1
    if not args.solo_exportar:
        for palabra in sorted(frecuencias):
            if frecuencias[palabra] >= 2 and palabra not in vocabulario:
                vocabulario[palabra] = dimension_vocabulario
                dimension_vocabulario += 1
        vocabulario.update({palabra: vocabulario[canon] for palabra, canon in alias.items() if canon in vocabulario})

    x_train, y_train = preparar(entrenamiento, vocabulario)
    x_val, y_val = preparar(validacion, vocabulario)
    if args.solo_exportar:
        modelo = anterior
    else:
        modelo = crear_modelo(dimension_vocabulario)
        if args.continuar:
            for previo, nuevo in zip(anterior.layers, modelo.layers, strict=True):
                pesos = previo.get_weights()
                if isinstance(nuevo, EmbeddingConPosicion):
                    ampliado = nuevo.get_weights()[0]
                    ampliado[: len(pesos[0])] = pesos[0]
                    pesos[0] = ampliado
                nuevo.set_weights(pesos)
            modelo.optimizer.learning_rate.assign(0.00045)
        print(f"Entrenamiento: {len(x_train)}. Validación sintética: {len(x_val)}. Evaluación escrita: {len(reservadas)}.", flush=True)
        modelo.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=192, epochs=args.epocas, verbose=2,
            callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        )

    metricas, registros, comparacion = {}, {}, {}
    for nombre, ejemplos in pruebas.items():
        entrada, etiquetas = preparar(ejemplos, vocabulario)
        probabilidades = modelo.predict(entrada, verbose=0)
        metricas[f"evaluacion_{nombre}"], predicciones = evaluar(probabilidades, etiquetas)
        clave = "predicciones" if nombre == "escrita" else f"predicciones_{nombre}"
        registros[clave] = detalle(ejemplos, predicciones)
        if args.continuar:
            entrada_anterior, _ = preparar(ejemplos, vocabulario_anterior)
            previas, _ = evaluar(anterior.predict(entrada_anterior, verbose=0), etiquetas)
            comparacion[nombre] = {"anterior": previas, "nuevo": metricas[f"evaluacion_{nombre}"]}
    if comparacion:
        total_anterior = sum(fila["anterior"]["exacta"] * fila["anterior"]["ejemplos"] for fila in comparacion.values())
        total_nuevo = sum(fila["nuevo"]["exacta"] * fila["nuevo"]["ejemplos"] for fila in comparacion.values())
        contexto = comparacion["contexto"]
        if total_nuevo <= total_anterior or contexto["nuevo"]["exacta"] < contexto["anterior"]["exacta"]:
            print(json.dumps(comparacion, indent=2))
            parser.exit(1, "Se conserva el modelo anterior porque el ajuste no superó la comparación.\n")
    metricas_val, _ = evaluar(modelo.predict(x_val, verbose=0), y_val)
    entrada_exportacion, _ = preparar(reservadas, vocabulario)
    lector = keras.Model(modelo.input, [modelo.get_layer(f"atencion_{nombre}").output for nombre in CABEZAS])
    for pesos in lector.predict(entrada_exportacion, verbose=0):
        np.testing.assert_allclose(pesos.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_array_equal(pesos[entrada_exportacion == 0], 0.0)
    PUBLICO.mkdir(parents=True, exist_ok=True)
    probs_exportacion = modelo.predict(entrada_exportacion, verbose=0)
    diferencia = exportar(modelo, entrada_exportacion, probs_exportacion, PUBLICO / "transformer.onnx")
    if not args.solo_exportar:
        modelo.save(archivo_keras, include_optimizer=False)
    (PUBLICO / "vocab.json").write_text(json.dumps({
        "version": 1,
        "longitud": LONGITUD,
        "tokens": vocabulario,
        "etiquetas": dict(zip(CABEZAS, ETIQUETAS, strict=True)),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    informe = {
        "entrenamiento": len(x_train),
        "parametros": modelo.count_params(),
        "validacion_sintetica": metricas_val,
        **metricas,
        "diferencia_maxima_onnx": diferencia,
        "comparacion_anterior": comparacion,
        **registros,
    }
    (RAIZ / "evaluacion_resultados.json").write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in informe.items() if not k.startswith("predicciones")}, indent=2))


if __name__ == "__main__":
    main()
