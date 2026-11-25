"""
consulta.py - Ruta del asistente (dueño: TOMÁS). Esqueleto creado por Javier el día 1.

POST /api/consulta recibe lo que entendió el transformer (en el navegador) y devuelve la
respuesta del chat, armada con datos reales de la tabla `eventos`. El formato del cuerpo lo
define Tomás, porque solo lo usa su propio chat (web/public/chat/chat.js).

Para leer eventos usar db.listar_eventos(desde, hasta, clase, camara, limite): funciona igual
con PostgreSQL y con SQLite local. Las fechas están en UTC; "hoy"/"ayer" se calculan en
hora de Colombia (zoneinfo.ZoneInfo("America/Bogota")).
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/consulta", tags=["consulta (Tomás)"])


@router.post("")
def consultar():
    return JSONResponse(status_code=501, content={"detalle": "Pendiente: esta ruta la implementa Tomás."})
