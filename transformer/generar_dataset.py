"""Genera preguntas etiquetadas sobre los eventos de DetAny3D."""

import random
import re
import unicodedata

CLASES = {
    "animales": ("animales", "animal", "fauna", "bichos", "especies", "todos los animales"),
    "bovino": ("vacas", "vaca", "ganado", "res", "reses", "bovinos", "bovino", "toros", "terneros"),
    "caballo": ("caballos", "caballo", "equinos", "equino", "yeguas", "yegua", "potros"),
    "gato": ("gatos", "gato", "felinos", "felino", "gatitos", "gatito", "michis", "gata"),
    "perro": ("perros", "perro", "caninos", "canino", "perritos", "canes", "perra"),
}
PERIODOS = {
    "sin_periodo": ("", "en el historial", "hasta ahora"),
    "hoy": ("hoy", "en el día de hoy", "durante el día de hoy", "en lo que va del día", "este día", "del día", "en la jornada de hoy"),
    "ayer": ("ayer", "el día de ayer", "durante el día de ayer", "en la jornada de ayer", "el día anterior", "la jornada anterior", "del día anterior"),
    "ultima_hora": ("en la última hora", "hace menos de una hora", "en los últimos 60 minutos", "en la hora pasada", "en los últimos sesenta minutos", "durante la última hora", "desde hace una hora"),
    "esta_semana": ("esta semana", "en lo que va de la semana", "desde el lunes", "en la semana actual", "en lo que va de semana"),
}
PLANTILLAS = {
    "contar": (
        "¿Cuántos registros de {animal} hubo {periodo}?",
        "Dame el número de apariciones de {animal} {periodo}.",
        "Total de {animal} detectados {periodo}.",
        "¿Cuántos {animal} pasaron {periodo}?",
        "Cuenta los eventos de {animal} {periodo}.",
        "{periodo}, ¿qué cantidad de {animal} se vio?",
        "¿Se vio algún ejemplar de {animal} {periodo}?",
        "Número de detecciones de {animal} {periodo}.",
        "¿Cuál es el conteo de {animal} {periodo}?",
        "¿Cuántas veces nos apareció {animal} {periodo}?",
        "Quisiera saber cuántos {animal} quedaron registrados {periodo}.",
        "{periodo}, saca el total de {animal} que alcanzamos a ver.",
        "Dime cuántos {animal} salieron en el video {periodo}.",
        "Dame la cifra de detecciones de {animal} {periodo}.",
    ),
    "listar": (
        "¿Qué {animal} pasaron {periodo}?",
        "Muéstrame la lista de {animal} {periodo}.",
        "Quiero ver los registros de {animal} {periodo}.",
        "Dame el historial de {animal} {periodo}.",
        "¿Cuáles fueron los avistamientos de {animal} {periodo}?",
        "Enumera los eventos de {animal} {periodo}.",
        "{periodo}, dime qué {animal} vio la cámara.",
        "Lista las detecciones de {animal} {periodo}.",
        "Pásame el listado de avistamientos de {animal} {periodo}.",
        "Enséñame lo que quedó registrado de {animal} {periodo}.",
        "¿Cuáles {animal} quedaron grabados {periodo}?",
        "Déjame revisar las apariciones de {animal} {periodo}.",
        "{periodo}, abre el historial de detecciones de {animal}.",
        "Pon en pantalla los eventos de {animal} {periodo}.",
    ),
    "ultimo_avistamiento": (
        "¿Cuándo se vio por última vez un ejemplar de {animal} {periodo}?",
        "Dame la detección más reciente de {animal} {periodo}.",
        "Último avistamiento de {animal} {periodo}.",
        "¿A qué hora registraron por última vez {animal} {periodo}?",
        "¿Cuál fue el último evento de {animal} {periodo}?",
        "Muéstrame el registro más nuevo de {animal} {periodo}.",
        "¿Qué hora tiene la aparición más reciente de {animal} {periodo}?",
        "{periodo}, dime cuándo pasó el último ejemplar de {animal}.",
        "¿A qué horas vieron el último animal de la clase {animal} {periodo}?",
        "Dime a qué hora apareció el último de los {animal} {periodo}.",
        "{periodo}, ubica la última vez que se registraron {animal}.",
        "El evento más reciente que figure de {animal} {periodo}.",
        "¿A qué hora pasó el más reciente de los {animal} {periodo}?",
        "¿Cuándo tuvo lugar la última aparición de {animal} {periodo}?",
    ),
    "resumen": (
        "Dame un resumen de las detecciones de {animal} {periodo}.",
        "¿Cómo estuvo la actividad de {animal} {periodo}?",
        "Haz un informe general de {animal} {periodo}.",
        "Necesito un reporte de avistamientos de {animal} {periodo}.",
        "{periodo}, resume lo que registró la cámara sobre {animal}.",
        "Balance de la actividad de {animal} {periodo}.",
        "Sintetiza qué pasó con {animal} {periodo}.",
        "Quiero un vistazo general a los eventos de {animal} {periodo}.",
        "Cuéntame a grandes rasgos qué vio la cámara de {animal} {periodo}.",
        "¿Cómo ha estado la actividad de los {animal} {periodo}?",
        "Hazme un recuento general de las apariciones de {animal} {periodo}.",
        "Dame el panorama de lo que pasó con los {animal} {periodo}.",
        "Necesito una visión general de {animal} que se vieron {periodo}.",
        "{periodo}, describe en general lo que registramos de {animal}.",
    ),
    "hora_frecuente": (
        "¿En qué hora hubo más detecciones de {animal} {periodo}?",
        "Dime la hora pico de los avistamientos de {animal} {periodo}.",
        "¿A qué hora pasaron más {animal} {periodo}?",
        "Identifica la franja con más registros de {animal} {periodo}.",
        "¿A qué hora se concentran las apariciones de {animal} {periodo}?",
        "Hora más frecuente de detecciones de {animal} {periodo}.",
        "¿Qué hora tuvo el mayor número de {animal} {periodo}?",
        "{periodo}, ¿cuándo hubo el pico de actividad de {animal}?",
        "¿En qué momento se vieron más {animal} {periodo}?",
        "¿En qué horario se registró la mayor cantidad de {animal} {periodo}?",
        "Indica a qué horas hubo más apariciones de {animal} {periodo}.",
        "¿Cuál fue el momento con mayor actividad de {animal} {periodo}?",
        "{periodo}, dime a qué hora se vieron {animal} con más frecuencia.",
        "Encuentra el horario que concentró más registros de {animal} {periodo}.",
    ),
}
FUERA_DE_DOMINIO = (
    "Hola, ¿cómo estás?", "¿Quién eres?", "¿Qué clima hará mañana?",
    "Dime la capital de Francia.", "¿Cuánto cuesta un perro?", "¿Qué comen los gatos?",
    "¿Cómo se cuida un caballo?", "Elabora una receta.", "Dime un chiste.",
    "¿Qué tal estuvo el partido ayer?", "Recomiéndame una película.",
    "Escribe un poema sobre una vaca.", "¿Dónde está la carretera?",
    "¿Cuál es mi contraseña?", "¿Qué hora es en Colombia?", "Pon música.",
    "Quiero cambiar el color de la página.", "Traduce perro al inglés.",
    "¿Cómo configuro la cámara?", "¿Cómo funciona un transformer?",
    "¿Por qué ladran los perros de noche?", "¿Dónde compro comida para gatos?",
    "¿Qué raza es ese caballo?", "¿Cuál es el peso de una vaca adulta?",
    "¿Puedes decirme qué hay en esta foto?", "¿Cuántos carros pasaron ayer?",
    "Lista los vehículos de hoy.", "Resumen de las noticias de esta semana.",
    "¿Cuándo fue el último partido de fútbol?", "¿Cuál es la hora pico del tráfico?",
    "¿Hubo alertas hoy?", "¿Qué alarma se activó ayer?",
    "Compara las vacas y los caballos hoy.", "¿Hubo más gatos o perros ayer?",
    "Muéstrame el historial de ventas.", "Cuenta los peatones de esta semana.",
    "¿Cómo puedo alimentar a un bovino?", "Dime la hora de la última entrega.",
    "Detecta animales en esta imagen.", "Elimina todos los eventos.",
    "¿Cuántos avistamientos habrá mañana?", "Enciende la cámara del patio.",
    "Incluye un informe del clima.", "Lista las especies de felinos en el mundo.",
    "¿Qué fecha es hoy?", "Quiero ver los goles del último encuentro.",
    "Guarda este video de caballos.", "Anota que hoy pasó un perro.",
)
TEMAS_AJENOS = (
    "carros", "autobuses", "peatones", "camiones", "noticias", "pedidos",
    "llamadas", "alarmas", "goles", "mensajes", "paquetes", "bicicletas",
)
PLANTILLAS_AJENAS = (
    "¿Cuántos {tema} hubo {periodo}?", "¿Cuáles {tema} pasaron {periodo}?",
    "Lista los registros de {tema} {periodo}.", "Dame un resumen de {tema} {periodo}.",
    "¿A qué hora hubo más {tema} {periodo}?", "Dime cuándo fue lo último de {tema} {periodo}.",
)
PLANTILLAS_AJENAS_CON_ANIMALES = (
    "¿Qué tipo de alimento necesitan los {animal}?",
    "¿Cada cuánto se vacuna a los {animal}?",
    "Recomiéndame una veterinaria para {animal}.",
    "¿Qué razas de {animal} existen?",
    "¿Dónde puedo adoptar o comprar {animal}?",
    "Quiero un listado de alimentos para {animal}.",
)
PLANTILLAS_GENERALES = {
    "contar": ("¿Cuántos eventos hubo {periodo}?", "¿Cuántas detecciones se registraron {periodo}?"),
    "listar": ("¿Qué se detectó {periodo}?", "Enséñame las detecciones {periodo}."),
    "ultimo_avistamiento": (
        "¿Cuándo fue la última detección {periodo}?", "Último registro {periodo}.",
        "¿Cuál fue el último animal registrado {periodo}?",
    ),
    "resumen": (
        "Dame un informe del sistema {periodo}.", "Resume las detecciones {periodo}.",
        "Cuéntame en general qué se registró {periodo}.",
    ),
    "hora_frecuente": (
        "¿Cuál fue la hora con más detecciones {periodo}?", "Hora pico de detecciones {periodo}.",
        "¿En qué momento se acumularon más avistamientos {periodo}?",
    ),
}
INTENCIONES = (*PLANTILLAS, "fuera_de_dominio")

GRUPOS_LEXICOS = (
    ("cuantos", "cuantas"),
    ("conteo", "cuenta"),
    ("numero", "cifra", "cantidad"),
    ("total", "acumulado"),
    ("ultimo", "ultima", "ultimos", "ultimas"),
    ("nuevo", "nueva", "nuevos", "nuevas"),
    ("lista", "listado", "listados", "listas"),
    ("cuales", "cual"),
    ("deteccion", "detecciones"),
    ("avistamiento", "avistamientos"),
    ("aparicion", "apariciones"),
    ("registro", "registros"),
    ("resumen", "sintesis"),
    ("general", "generales"),
    ("franja", "horario"),
    ("hora", "horas"),
    ("minuto", "minutos"),
    ("60", "sesenta"),
    ("alerta", "alertas", "alarma", "alarmas"),
    ("evento", "eventos"),
    ("anterior", "previo", "previa"),
    ("autobuses", "buses", "autobus", "bus"),
)

CONTRASTES = {
    "contar": (
        "¿Llevamos muchas detecciones de {animal} {periodo}?",
        "¿Se llegó a ver {animal} {periodo}?",
        "¿Cuál es el total de veces que se vio {animal} {periodo}?",
        "{periodo}, ¿cuántos {animal} consiguió registrar la cámara?",
        "Necesito saber el número de {animal} que aparecen en los registros {periodo}.",
        "Conteo de {animal} en los registros {periodo}.",
    ),
    "listar": (
        "¿Qué registros de {animal} quedaron {periodo}?",
        "Dime cuáles de los {animal} se detectaron {periodo}.",
        "Saca el listado con las apariciones de {animal} {periodo}.",
        "¿Me muestras las apariciones que hubo de {animal} {periodo}?",
        "{periodo}, quiero revisar lo que se registró de {animal}.",
        "Dame el detalle de cada aparición de {animal} {periodo}.",
    ),
    "ultimo_avistamiento": (
        "De los {animal} {periodo}, ¿cuál es el registro más reciente?",
        "Ubícame la última aparición que quedó de {animal} {periodo}.",
        "Me interesa saber cuándo ocurrió la detección más nueva de {animal} {periodo}.",
        "¿Cuándo registraron de último a {animal} {periodo}?",
        "{periodo}, dime a qué hora apareció {animal} por última vez.",
        "Necesito saber la hora del último registro de {animal} {periodo}.",
    ),
    "resumen": (
        "Haz una síntesis de lo que vio la cámara de {animal} {periodo}.",
        "Descríbeme en términos generales qué se registró de {animal} {periodo}.",
        "¿Cómo estuvo el movimiento de {animal} {periodo}?",
        "Quiero el panorama de apariciones de {animal} {periodo}.",
        "En general, ¿qué pasó con {animal} {periodo}?",
        "{periodo}, dame un balance de lo que se observó de {animal}.",
    ),
    "hora_frecuente": (
        "¿Cuál fue el pico de detecciones de {animal} {periodo}?",
        "¿En qué franja horaria se juntaron más avistamientos de {animal} {periodo}?",
        "¿A qué horas se registraron más {animal} {periodo}?",
        "{periodo}, ubica la hora que concentró más detecciones de {animal}.",
        "De todos los {animal} {periodo}, dime el horario de mayor actividad.",
        "¿Cuál fue la franja más movida para {animal} {periodo}?",
    ),
}

PLANTILLAS_CONTRASTE = (
    "De los registros de {animal} {periodo}, necesito {pedido}.",
    "Sobre {animal} {periodo}, dime {pedido}.",
)
PEDIDOS = {
    "contar": ("el total", "cuántos hubo", "el número de apariciones"),
    "listar": ("la lista", "cuáles quedaron registrados", "el listado completo"),
    "ultimo_avistamiento": ("la última aparición", "cuál fue el último registro", "el más reciente"),
    "resumen": ("un balance", "el panorama general", "el resumen de actividad"),
    "hora_frecuente": ("la hora pico", "la hora con más apariciones", "en qué momento se vieron más"),
}


def tokens(texto):
    normalizado = unicodedata.normalize("NFD", texto.lower())
    normalizado = "".join(letra for letra in normalizado if not unicodedata.combining(letra))
    return re.findall(r"[a-z0-9]+", normalizado)


def alias_lexicos():
    alias = {variante: clase for clase, nombres in CLASES.items() for variante in (*nombres, clase) if " " not in variante}
    for grupo in GRUPOS_LEXICOS:
        alias.update(dict.fromkeys(grupo, grupo[0]))
    return alias


def codificar(texto, vocabulario, longitud):
    ids = [vocabulario.get(token, 1) for token in tokens(texto)[:longitud]] or [1]
    return ids + [0] * (longitud - len(ids))


def variante(texto, rng):
    palabras = tokens(texto)
    if rng.random() < 0.35:
        saludo = rng.choice((["por", "favor"], ["oye"], ["porfa"], ["me", "ayudas"]))
        palabras = saludo + palabras if rng.random() < 0.75 else palabras + saludo
    if rng.random() < 0.25:
        candidatas = [i for i, palabra in enumerate(palabras) if len(palabra) > 4]
        if candidatas:
            i = rng.choice(candidatas)
            palabra = palabras[i]
            j = rng.randrange(1, len(palabra) - 1)
            palabras[i] = palabra[:j] + palabra[j + 1] + palabra[j] + palabra[j + 2 :]
    return " ".join(palabras)


def generar(semilla=42):
    rng = random.Random(semilla)
    filas = {}
    for intencion, plantillas in PLANTILLAS.items():
        for plantilla in (*plantillas, *CONTRASTES[intencion]):
            for clase, animales in CLASES.items():
                for periodo, periodos in PERIODOS.items():
                    for animal in animales:
                        texto = plantilla.format(animal=animal, periodo=rng.choice(periodos))
                        etiqueta = (intencion, clase, periodo)
                        filas[" ".join(tokens(texto))] = etiqueta
                        filas[variante(texto, rng)] = etiqueta
        for plantilla in PLANTILLAS_CONTRASTE:
            for clase, animales in CLASES.items():
                for periodo, periodos in PERIODOS.items():
                    for pedido in PEDIDOS[intencion]:
                        texto = plantilla.format(animal=rng.choice(animales), periodo=rng.choice(periodos), pedido=pedido)
                        filas[" ".join(tokens(texto))] = (intencion, clase, periodo)
        for plantilla in PLANTILLAS_GENERALES[intencion]:
            for periodo, periodos in PERIODOS.items():
                for frase in periodos:
                    texto = plantilla.format(periodo=frase)
                    etiqueta = (intencion, "animales", periodo)
                    filas[" ".join(tokens(texto))] = etiqueta
                    filas[variante(texto, rng)] = etiqueta
    textos_ajenos = list(FUERA_DE_DOMINIO) + [
        "¿Hay alertas de animales en la vía?",
        "¿Puedo configurar alarmas para cuando aparezca un animal?",
        "¿Qué alertas se registraron durante este día?",
        "¿Qué animales activaron una alerta durante la última hora?",
        "Avísame por correo si tienes alarmas.",
    ]
    for plantilla in PLANTILLAS_AJENAS:
        for tema in TEMAS_AJENOS:
            for periodos in PERIODOS.values():
                textos_ajenos.append(plantilla.format(tema=tema, periodo=rng.choice(periodos)))
    for primero in ("vacas", "caballos", "perros", "gatos"):
        for segundo in ("ganado", "equinos", "canes", "felinos"):
            for periodos in PERIODOS.values():
                periodo = rng.choice(periodos)
                textos_ajenos.extend((
                    f"Compara los registros de {primero} con los de {segundo} {periodo}.",
                    f"¿Se vieron más {primero} o {segundo} {periodo}?",
                ))
    for plantilla in PLANTILLAS_AJENAS_CON_ANIMALES:
        for clase, animales in CLASES.items():
            if clase != "animales":
                for animal in animales:
                    textos_ajenos.append(plantilla.format(animal=animal))
    for texto in textos_ajenos:
        etiqueta = ("fuera_de_dominio", "animales", "sin_periodo")
        filas[" ".join(tokens(texto))] = etiqueta
        for _ in range(4):
            filas[variante(texto, rng)] = etiqueta

    ejemplos = list(filas.items())
    rng.shuffle(ejemplos)
    return ejemplos


def generar_refuerzo(semilla=43):
    rng = random.Random(semilla)
    contrastes = {
        "contar": (
            "¿Se alcanzó a ver al menos un ejemplar de {animal} {periodo}?",
            "¿Cuántas veces aparece {animal} en lo que grabó la cámara {periodo}?",
            "{periodo}, ¿cuánto suma lo registrado de {animal}?",
            "Sobre {animal} {periodo}, necesito saber cuántos hubo en total.",
        ),
        "listar": (
            "¿Qué detecciones hubo de {animal} {periodo}?",
            "Dime qué {animal} aparecen en el historial de la cámara {periodo}.",
            "{periodo}, enséñame lo que detectó la cámara de {animal}.",
            "Sobre {animal} {periodo}, necesito ver el detalle de cada detección.",
        ),
        "ultimo_avistamiento": (
            "¿A qué hora anotaron la última detección de {animal} {periodo}?",
            "¿En qué momento se vio por última vez {animal} {periodo}?",
            "{periodo}, ubica el último de los {animal} que quedaron registrados.",
            "Sobre {animal} {periodo}, necesito únicamente el avistamiento más reciente.",
        ),
        "resumen": (
            "En pocas palabras, ¿cómo estuvo la actividad de {animal} {periodo}?",
            "¿Qué balance haces de lo visto de {animal} en cámara {periodo}?",
            "{periodo}, dame una vista general de lo registrado de {animal}.",
            "Sobre {animal} {periodo}, necesito una síntesis de toda la actividad.",
        ),
        "hora_frecuente": (
            "¿A qué hora anotaron más detecciones de {animal} {periodo}?",
            "¿En qué momento se vio el mayor número de {animal} {periodo}?",
            "{periodo}, ubica el horario que concentró más registros de {animal}.",
            "Sobre {animal} {periodo}, necesito la hora de mayor actividad.",
        ),
    }
    sesiones = {
        "sin_periodo": ("", "en todo el historial", "hasta ahora"),
        "hoy": ("hoy", "en la jornada actual", "en lo que va del día"),
        "ayer": ("ayer", "el día pasado", "durante el día anterior", "en el día previo"),
        "ultima_hora": ("en la última hora", "en la hora previa", "en los pasados 60 minutos", "desde hace una hora"),
        "esta_semana": ("esta semana", "desde el comienzo de la semana", "en lo que va de semana"),
    }
    generales = {
        "contar": ("¿En cuánto va el total de detecciones {periodo}?", "¿Cuántas apariciones contó la cámara {periodo}?"),
        "listar": ("¿Qué animales quedaron anotados por la cámara {periodo}?", "Muéstrame lo que se vio {periodo}."),
        "ultimo_avistamiento": ("¿Qué fue lo último que vio la cámara {periodo}?", "¿Cuál fue el último animal detectado {periodo}?"),
        "resumen": ("Quiero una síntesis de la actividad captada por la cámara {periodo}.", "Dame una visión general de lo que se detectó {periodo}."),
        "hora_frecuente": ("¿A qué hora se registraron más apariciones {periodo}?", "¿Cuándo se concentraron más detecciones {periodo}?"),
    }
    for intencion, plantillas in contrastes.items():
        for periodo, fechas in sesiones.items():
            for fecha in fechas:
                for plantilla in generales[intencion]:
                    yield plantilla.format(periodo=fecha), (intencion, "animales", periodo)
                for clase, animales in CLASES.items():
                    for animal in rng.sample(animales, 3):
                        for plantilla in plantillas:
                            yield plantilla.format(animal=animal, periodo=fecha), (intencion, clase, periodo)

    etiqueta_ajena = ("fuera_de_dominio", "animales", "sin_periodo")
    for animales in CLASES.values():
        for animal in animales:
            for plantilla in (
                "¿Cuántas veces al día deben comer los {animal}?",
                "¿Cuándo pusieron la última vacuna a los {animal}?",
                "Busca una lista de veterinarios que atiendan {animal}.",
                "¿Cómo se alimenta a los {animal} de la finca?",
                "¿Qué alerta se dispara si aparece un ejemplar de {animal}?",
            ):
                yield plantilla.format(animal=animal), etiqueta_ajena
    for periodo in ("hoy", "ayer", "en los pasados 60 minutos", "esta semana"):
        for tema in ("buses", "camiones", "noticias", "goles", "entregas", "taxis"):
            for plantilla in (
                "¿A qué hora pasó lo último de {tema} {periodo}?",
                "¿A qué hora hubo más {tema} {periodo}?",
                "Quiero un recuento de {tema} {periodo}.",
            ):
                yield plantilla.format(tema=tema, periodo=periodo), etiqueta_ajena
