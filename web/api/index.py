"""
index.py - App FastAPI (dueño: Javier).

Local:   cd web ; ..\\.venv\\Scripts\\python.exe -m uvicorn app:app --reload --port 8000
         y abrir http://localhost:8000
Vercel:  Root Directory = web/. Vercel encuentra la app en web/app.py, sirve public/ desde su
         CDN y esta app atiende /api/*. Variable de entorno: DATABASE_URL (PostgreSQL de River).
"""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import consulta, db, eventos

app = FastAPI(title="Animales en la vía - API", version="1.0")
app.include_router(eventos.router)
app.include_router(consulta.router)     # ruta de Tomás (registrada desde el día 1)


@app.get("/api/salud")
def salud():
    return {"ok": True, "base": db.motor()}


# En local, FastAPI también sirve la página. En Vercel NO: su CDN sirve public/ y la
# documentación pide no montarla con app.mount().
PUBLICO = Path(__file__).resolve().parents[1] / "public"
if PUBLICO.is_dir() and not os.environ.get("VERCEL"):
    app.mount("/", StaticFiles(directory=PUBLICO, html=True), name="publico")
