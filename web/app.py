"""
app.py - Punto de entrada que Vercel detecta para FastAPI (busca un `app` en app.py).

La app vive en api/index.py; aquí solo se importa.
Local:  cd web ; ..\\.venv\\Scripts\\python.exe -m uvicorn app:app --port 8000
"""
from api.index import app  # noqa: F401
