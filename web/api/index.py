"""
index.py - App FastAPI (dueño: Javier).

Local:   cd web ; ..\\.venv\\Scripts\\python.exe -m uvicorn api.index:app --reload --port 8000
         y abrir http://localhost:8000
Vercel:  Root Directory = web/. Vercel sirve public/ como archivos estáticos y esta app
         atiende /api/*. Variable de entorno: DATABASE_URL (PostgreSQL de River).
"""
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


# En local, FastAPI también sirve la página. En Vercel lo hace su CDN desde public/.
PUBLICO = Path(__file__).resolve().parents[1] / "public"
if PUBLICO.is_dir():
    app.mount("/", StaticFiles(directory=PUBLICO, html=True), name="publico")
