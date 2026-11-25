"""Consulta los avistamientos que guarda la CNN mediante la API de eventos."""

import re
import unicodedata
from collections import Counter
from datetime import datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import db

router = APIRouter(prefix="/api/consulta", tags=["consulta"])
BOGOTA = ZoneInfo("America/Bogota")
Periodo = Literal["sin_periodo", "hoy", "ayer", "ultima_hora", "esta_semana"]
Intencion = Literal[
    "contar", "listar", "ultimo_avistamiento", "resumen", "hora_frecuente", "fuera_de_dominio",
    "identidad",
]
Clase = Literal["animales", "bovino", "caballo", "gato", "perro"]
NOMBRES = {"animales": "animales", "bovino": "bovinos", "caballo": "caballos", "gato": "gatos", "perro": "perros"}
RECHAZO = "Solo te puedo contar lo que quedó registrado por la cámara."

ACCIONES = frozenset("""
    cuanta cuantas cuanto cuantos contar cuenta conteo cantidad total acumulado cifra numero
    lista listas listar listado listados enumera muestra muestrame ensename mostrar ver revisar
    cuales cual que detalle historial ultimas ultimos ultima ultimo reciente recientes cuando
    resume resumen informe balance recuento panorama sintesis general generales reporte
    hora horas franja horario pico frecuente frecuencia momento concentracion
    abre busca dejame dime dame elige encuentra haz hazme interesa llevamos marca pasame ponme
    registra registrar saca sacame saber ubica ubicame cuentame describeme vio ver visto
    aparecio quedaron como nuevo nueva actividad movimiento viste
""".split())
AVISTAMIENTOS = frozenset("""
    animal animales bovino bovinos vaca vacas ganado res reses toro toros terneros
    caballo caballos yegua yeguas equino equinos potro potros
    gato gatos gata felino felinos gatito gatitos michis
    perro perros perra canes canino caninos perritos fauna bichos
    camara camaras avistamiento avistamientos deteccion detecciones registro registros evento eventos
    aparicion apariciones video videos grabacion grabaciones detecto detectamos detectaron detectados detectadas
    registrado registrada registrados registradas
""".split())
VOCABULARIO = ACCIONES | AVISTAMIENTOS | frozenset("""
    a al ante con de del desde durante el ella en entre esa ese esta estas este estos ha han hasta
    la las le les lo los me mi por que se si su te tu un una uno unas unos y o
    hoy ayer dia dias jornada semana lunes anterior previa previo pasado pasada pasados
    actual ahora reciente ultimo ultima minutos minuto 60 sesenta menos mas mayor maxima maximo
    hay hubo habia aparecen aparece aparecieron aparecio aparecer viendo vio vimos visto vieron vea
    paso pasan pasaron salio salen salieron quedaron quedo registra registrado registrada registrados
    registradas registraron registro detectamos detecto detectados detectadas detectaron detectado detectada
    figuran figura grabo grabados grabadas grabaron capto captados captadas captaron anotaron anoto
    quedo ocurrido ocurrio ocurren ocurrio actividad movimiento ejemplar ejemplares cualquier algun alguna
    cada todo todos toda todas poco pocas palabras vario varios varias veces vez juntos conjunto
    dame dime quiero quisiera necesito saber pon pantalla saca sacame busca haz hazme haznos
    cuentame describeme alcanza alcanzo alcanzamos alcanzo quedo toca favor porfa oye ayudas puedes
    puede podria podrias gracias porfis como cuantas cuantos suma llevo llevamos seguidos visibles
    produjo produjeron ubicada ubica ubicame marca identifica senala concentraron concentro junta juntaron
    detecta detectan registra registran suelen suele mira para aqui alli solo solamente inicio comienzo
    segun captura capturas fuente fuentes pestana webcam colombia bogota perdona vio vi vio
    activa acumulo das dejaron dentro es estado estuvo fue grandes hace hemos llego movida movio
    muchas muestras nos rapido rasgos registramos salir sido sobre son terminos tramo tuvo va vision vista
    tenga estuvieron interesa favor manera simplemente especialmente aproximadamente detectar
    ey dices pasas cuentas grabamos haces salido nada
""".split())
EVIDENCIA = frozenset("""
    avistamiento avistamientos deteccion detecciones registro registros evento eventos historial
    grabacion grabaciones video videos aparicion apariciones detectar detecto detectaron detectamos
    detectado detectados detectada detectadas registra registrado registrados registrada registradas
    registraron registrar registramos quedaron quedo vio vimos visto vieron viendo vea viste pasaron
    paso pasan pasado salio salieron salir aparecen aparecio aparecieron capto captados
    grabados grabamos anoto anotaron figuran ver concentro salido
""".split())
OPERACIONES = frozenset("""
    cuanta cuantas cuanto cuantos contar conteo total numero cantidad acumulado lista listar listado resume resumen balance
    ultimo ultima ultimos ultimas reciente recientes pico franja
""".split())
TIEMPOS = {"hoy", "ayer", "semana", "anterior", "lunes", "minutos", "hora", "horas", "dia", "jornada"}
CONTEXTO_TEMPORAL = {"actividad", "movimiento", "mas", "mayor", "concentracion"}


class Consulta(BaseModel):
    pregunta: str = Field(min_length=1, max_length=300)
    intencion: Intencion
    clase: Clase
    periodo: Periodo


def sobre_avistamientos(texto: str) -> bool:
    sin_tildes = unicodedata.normalize("NFD", texto.lower())
    sin_tildes = "".join(letra for letra in sin_tildes if not unicodedata.combining(letra))
    if re.search(r"[^a-z0-9\s¿?¡!.,;:()\-]", sin_tildes):
        return False
    palabras = re.findall(r"[a-z0-9]+", sin_tildes)
    vocabulario = set(palabras)
    contexto = bool(vocabulario & TIEMPOS) and bool(vocabulario & CONTEXTO_TEMPORAL)
    return (
        1 <= len(palabras) <= 24
        and vocabulario <= VOCABULARIO
        and bool(vocabulario & ACCIONES)
        and bool(vocabulario & AVISTAMIENTOS)
        and (contexto or bool(vocabulario & (EVIDENCIA | OPERACIONES)))
        and not sin_tildes.lstrip("¿¡ ").startswith(("que es ", "que son "))
    )


def ventana(periodo: Periodo, ahora: datetime):
    medianoche = ahora.replace(hour=0, minute=0, second=0, microsecond=0)
    if periodo == "hoy":
        return medianoche, medianoche + timedelta(days=1)
    if periodo == "ayer":
        return medianoche - timedelta(days=1), medianoche
    if periodo == "ultima_hora":
        return ahora - timedelta(hours=1), ahora
    if periodo == "esta_semana":
        lunes = medianoche - timedelta(days=medianoche.weekday())
        return lunes, lunes + timedelta(days=7)
    return None, None


def cantidad(total):
    return "1 avistamiento" if total == 1 else f"{total} avistamientos"


def responder(texto, datos):
    return {"respuesta": texto, "datos": datos}


@router.post("")
def consultar(consulta: Consulta):
    intencion = consulta.intencion
    if intencion == "identidad":
        return responder("Soy Javiercito, el asistente de DetAny3D.", {"nombre": "Javiercito"})
    if intencion == "fuera_de_dominio" or not sobre_avistamientos(consulta.pregunta):
        return {"respuesta": RECHAZO, "datos": None}

    desde, hasta = ventana(consulta.periodo, datetime.now(BOGOTA))
    clase = None if consulta.clase == "animales" else consulta.clase
    # El límite de /api/eventos no aplica a los conteos del asistente.
    limite = {"ultimo_avistamiento": 1, "listar": 50}.get(intencion, 2**63 - 1)
    try:
        eventos = db.listar_eventos(desde=desde, hasta=hasta, clase=clase, limite=limite)
    except Exception as error:
        raise HTTPException(503, "No se pudo consultar el historial de avistamientos.") from error

    animal = NOMBRES[consulta.clase]
    if intencion == "contar":
        total = len(eventos)
        verbo = "quedó" if total == 1 else "quedaron"
        texto = f"En ese periodo {verbo} {cantidad(total)} de {animal}."
        return responder(texto, {"total": total})
    if not eventos:
        datos = {"total": 0, "por_clase": []} if intencion == "resumen" else []
        if intencion == "ultimo_avistamiento":
            datos = None
        return responder("No quedó ningún avistamiento en ese periodo.", datos)
    if intencion == "listar":
        campos = ("id", "clase", "inicio", "fin", "camara")
        datos = [{campo: evento[campo] for campo in campos} for evento in eventos]
        return responder(f"Encontré {cantidad(len(eventos))} de {animal}.", datos)
    if intencion == "ultimo_avistamiento":
        ultimo = eventos[0]
        local = ultimo["inicio"].astimezone(BOGOTA)
        datos = {campo: ultimo[campo] for campo in ("id", "clase", "inicio", "fin", "camara")}
        texto = f"El último avistamiento de {ultimo['clase']} fue el {local:%d/%m/%Y a las %H:%M}, hora de Colombia."
        return responder(texto, datos)
    if intencion == "resumen":
        conteos = Counter(evento["clase"] for evento in eventos)
        detalle = ", ".join(f"{total} de {NOMBRES[nombre]}" for nombre, total in sorted(conteos.items()))
        datos = {
            "total": len(eventos),
            "por_clase": [{"clase": nombre, "total": total} for nombre, total in sorted(conteos.items())],
        }
        texto = f"En ese periodo se registró un total de {cantidad(len(eventos))} ({detalle})."
        return responder(texto, datos)
    horas = Counter(evento["inicio"].astimezone(BOGOTA).hour for evento in eventos)
    maximo = max(horas.values())
    picos = sorted(hora for hora, total in horas.items() if total == maximo)
    detalle = ", ".join(f"{hora:02d}:00" for hora in picos)
    texto = f"La hora con más actividad tuvo {cantidad(maximo)} ({detalle}, hora de Colombia)."
    return responder(texto, [{"hora": hora, "total": maximo} for hora in picos])
