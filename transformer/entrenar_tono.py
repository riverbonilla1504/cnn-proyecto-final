"""Entrena desde cero el Transformer de identidad y tono de Javiercito."""

import argparse
import json
import random
import re
from pathlib import Path

from entrenar import (
    EmbeddingConPosicion, LONGITUD, PUBLICO, generar, keras, layers, np, ops, ort, tf, tf2onnx, tokens,
)

MENSAJE = re.compile(
    r"(?m)^[\u200e\u200f\ufeff]*\[?[\d/.:\- ,aApPmM\u00a0\u202f]+"
    r"(?:\]\s*| - )([^:\r\n]{1,90}): ([^\r\n]*)"
)
MARCADORES = ("bruto", "bueno", "pues")
GRUPO = re.compile(r"\b(?:bruto|bueno|pues)\b(?:[\s,¡!¿?.]+\b(?:bruto|bueno|pues)\b)*")
SALIDAS = ("consulta", "identidad", "fin", *MARCADORES)
IDENTIDAD = (
    "cómo te llamas", "cómo se llama usted", "cómo te llamás pues", "tú cómo te llamas",
    "cuál es tu nombre", "me dices tu nombre", "oye dime tu nombre", "cómo es que te llamas",
    "cómo te puedo llamar", "cómo te digo a ti", "cómo querés que te llame", "cómo te puedo decir",
    "qué nombre tienes", "tienes algún nombre", "tienes un nombre", "tu nombre porfa", "sabes tu propio nombre",
    "tú sabes cómo te llamas", "dime quién eres", "quién eres tú", "tú quién sos",
    "se puede saber quién eres", "con quién estoy hablando", "eres Javiercito",
    "tú eres Javiercito cierto", "te llamas Javiercito o cómo", "cuál era tu nombre otra vez",
    "eres Javiercito el asistente de DetAny3D", "tú eres el Javiercito de la página",
    "quién me contesta aquí", "quién está respondiendo acá", "con quién acabo de hablar",
    "con quién hablo en el chat", "qué nombre te pusieron", "cómo te pusieron a ti",
)
CONTRASTES = (
    "cómo se llama el perro", "dime el nombre del último gato", "cuál es el nombre del caballo",
    "quién es Javier", "quién hizo la cámara", "quién pasó por la cámara hoy",
    "cuál es el nombre de tu dueño", "cómo se llama tu dueño",
    "qué nombre aparece en el registro", "cuéntame sobre los perros que captaste",
    "qué pasó hoy", "cuál es la capital de Colombia", "quién es el presidente",
    "cómo te fue con los registros", "cómo estás", "quién canta esa canción",
    "Javiercito cuántos gatos aparecieron", "Javiercito dime qué viste ayer",
    "cuéntame un chiste Javiercito", "buenas Javiercito", "hola Javiercito",
    "quién eres y cuánto vale el dólar", "cómo te llamas y hazme la tarea",
    "eres Javiercito y qué presidente gobierna", "cómo pusieron de nombre al perro",
    "quién está manejando la cámara", "quién me contesta el correo",
)


def secuencias_de_javier(ruta):
    mensajes = MENSAJE.findall(ruta.read_text(encoding="utf-8-sig"))
    autores = {autor for autor, _ in mensajes if "javi" in autor.casefold()}
    if len(autores) != 1:
        raise ValueError("El archivo debe identificar a un único remitente como Javier.")
    autor = autores.pop()
    secuencias = []
    for remitente, mensaje in mensajes:
        if remitente != autor or mensaje.lstrip(" \u200e\u200f").startswith(("<", "http:", "https:")):
            continue
        secuencias.extend(tuple(tokens(grupo)) for grupo in GRUPO.findall(mensaje.casefold()))
    return secuencias


def entrenar(ruta):
    secuencias = secuencias_de_javier(ruta)
    if not secuencias:
        raise ValueError("No se encontraron secuencias de tono en los mensajes de Javier.")

    keras.utils.set_random_seed(61)
    rellenos = ("oye", "a ver", "parce", "una pregunta", "perdona", "dime una cosa", "al fin")
    azar = random.Random(61)
    ejemplos = []
    consultas = [*CONTRASTES, *(texto for texto, _ in generar()[:320])]
    for textos, etiqueta, variantes in ((IDENTIDAD, "identidad", 7), (consultas, "consulta", 1)):
        for texto in textos:
            palabras = tokens(texto)
            ejemplos.append((["pregunta", *palabras], etiqueta))
            for _ in range(variantes):
                relleno = tokens(azar.choice(rellenos))
                combinado = relleno + palabras if azar.random() < 0.8 else palabras + relleno
                ejemplos.append((["pregunta", *combinado], etiqueta))
    ejemplos.extend((["pregunta", *tokens(texto)], "consulta") for texto in CONTRASTES for _ in range(5))
    # Complemento explícito aportado por el usuario.
    for secuencia in [*secuencias, *[MARCADORES] * 20]:
        secuencia = secuencia[:3]
        for paso, salida in enumerate((*secuencia, "fin")):
            ejemplos.append((["tono", *secuencia[:paso]], salida))

    palabras = sorted({palabra for entrada, _ in ejemplos for palabra in entrada})
    vocabulario = {palabra: indice for indice, palabra in enumerate(("<pad>", "<unk>", *palabras))}
    entradas = np.zeros((len(ejemplos), LONGITUD), dtype=np.int32)
    etiquetas = np.array([SALIDAS.index(salida) for _, salida in ejemplos], dtype=np.int32)
    rng = np.random.default_rng(61)
    for indice, (entrada, _) in enumerate(ejemplos):
        ids = [vocabulario[palabra] for palabra in entrada[:LONGITUD]]
        if entrada[0] == "pregunta" and len(ids) > 3 and rng.random() < 0.25:
            ids[rng.integers(1, len(ids))] = vocabulario["<unk>"]
        entradas[indice, :len(ids)] = ids

    entrada = keras.Input(shape=(LONGITUD,), dtype="int32", name="tokens")
    mascara = ops.expand_dims(ops.not_equal(entrada, 0), axis=1)
    posiciones = EmbeddingConPosicion(len(vocabulario), 32)(entrada)
    atencion = layers.MultiHeadAttention(num_heads=4, key_dim=8)(posiciones, posiciones, attention_mask=mascara)
    x = layers.LayerNormalization()(posiciones + atencion)[:, 0, :]
    x = layers.Dense(48, activation="relu")(x)
    modelo = keras.Model(entrada, layers.Dense(len(SALIDAS), activation="softmax")(x))
    modelo.compile(optimizer=keras.optimizers.Adam(0.003), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    resultado = modelo.fit(entradas, etiquetas, epochs=32, batch_size=32, verbose=0)

    firma = (tf.TensorSpec((None, LONGITUD), tf.int32, name="tokens"),)

    @tf.function(input_signature=firma)
    def predecir(entrada):
        return {"probabilidades": modelo(entrada, training=False)}

    grafico, _ = tf2onnx.convert.from_function(predecir, input_signature=firma, opset=17)
    sesion = ort.InferenceSession(grafico.SerializeToString(), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(
        sesion.run(["probabilidades"], {"tokens": entradas[:16]})[0],
        modelo(entradas[:16], training=False).numpy(),
        atol=1e-5,
    )
    (PUBLICO / "tono.onnx").write_bytes(grafico.SerializeToString())
    (PUBLICO / "tono.json").write_text(json.dumps({
        "longitud": LONGITUD, "tokens": vocabulario, "salidas": SALIDAS, "marcadores": MARCADORES,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Secuencias del chat: {len(secuencias)}. Exactitud de entrenamiento: {resultado.history['accuracy'][-1]:.1%}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chat", type=Path)
    entrenar(parser.parse_args().chat)
