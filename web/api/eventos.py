"""
eventos.py - Rutas de eventos (dueño: Javier). Formato fijado en contrato/esquema.sql.

POST /api/eventos   guarda un evento (una aparición de un animal) -> {"id": 123}
GET  /api/eventos   lista eventos, más reciente primero (filtros opcionales)
"""
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from . import db

router = APIRouter(prefix="/api/eventos", tags=["eventos"])

Clase = Literal["bovino", "caballo", "gato", "perro"]
Camara = Literal["webcam_1", "video", "pestana", "simulada"]


class EventoNuevo(BaseModel):
    inicio: datetime
    fin: datetime
    camara: Camara = "webcam_1"
    clase: Clase
    confianza: float = Field(ge=0, le=1)
    confianza_max: float = Field(ge=0, le=1)
    n_frames: int = Field(ge=1)
    modo: Literal["rapido", "preciso"]
    miniatura_url: Optional[str] = None

    @model_validator(mode="after")
    def _coherente(self):
        if self.inicio.tzinfo is None or self.fin.tzinfo is None:
            raise ValueError("inicio y fin deben incluir zona horaria (ej. 2026-11-25T19:02:10Z)")
        if self.fin < self.inicio:
            raise ValueError("fin debe ser igual o posterior a inicio")
        if self.confianza > self.confianza_max + 1e-6:
            raise ValueError("confianza (promedio) no puede ser mayor que confianza_max")
        return self


@router.post("", status_code=201)
def crear_evento(evento: EventoNuevo):
    try:
        return {"id": db.insertar_evento(evento.model_dump())}
    except Exception as e:  # la base rechazó el evento o no está disponible
        raise HTTPException(status_code=503, detail=f"No se pudo guardar el evento: {e}") from e


@router.get("")
def listar(desde: Optional[datetime] = None, hasta: Optional[datetime] = None,
           clase: Optional[Clase] = None, camara: Optional[Camara] = None,
           limite: int = Query(50, ge=1, le=500)):
    return db.listar_eventos(desde=desde, hasta=hasta, clase=clase, camara=camara, limite=limite)
