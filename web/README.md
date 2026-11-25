# Web: CNN en el navegador + API de eventos

Página que corre la CNN propia directamente en el navegador con onnxruntime-web 1.31.1. Tiene 4 fuentes de imagen (cámara, video subido, pestaña compartida y foto), modo rápido o preciso y un umbral ajustable. Una API FastAPI guarda los eventos (una aparición de un animal = un evento) en una base de datos.

## Correr en local

Desde la raíz del repo, en PowerShell:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r web\requirements.txt uvicorn
cd web
..\.venv\Scripts\python.exe -m uvicorn api.index:app --reload --port 8000
```

Abrir http://localhost:8000. La cámara y la pestaña compartida solo funcionan en `localhost` o con HTTPS.

## Base de datos

- Sin la variable `DATABASE_URL`, la API usa SQLite local (`web/eventos_local.db`) automáticamente.
- Con PostgreSQL (la de River), define la variable antes de arrancar:

```powershell
$env:DATABASE_URL = "postgresql://..."
```

- En local la tabla `eventos` se crea sola a partir de `contrato/esquema.sql`.
- En Vercel hay que ejecutar ese esquema una vez con `psql`.

## Regenerar los ONNX

```powershell
.venv\Scripts\python.exe src\exportar_onnx.py
.venv\Scripts\python.exe src\verificar_onnx.py
```

Para probarlos en el navegador, abrir `/prueba_onnx.html` (compara las fotos de prueba con `esperado.json`).

## Desplegar en Vercel

- *Root Directory* = `web`.
- Variable de entorno `DATABASE_URL` con la cadena de PostgreSQL.
- Aviso: el despliegue todavía no se ha probado.

## Estructura y dueños

| Ruta | Dueño | Contenido |
| --- | --- | --- |
| `web/public/index.html`, `js/`, `css/` | Javier | Página de cámara (`modelo.js`, `eventos.js`, `app.js`, `estilos.css`) |
| `web/public/modelos/` | Javier | `cnn_rapido.onnx`, `cnn_preciso.onnx`, `modelo.json`, `esperado.json` |
| `web/public/chat/` | Tomás | Chat con el transformer (por ahora vacía) |
| `web/api/index.py`, `eventos.py`, `db.py` | Javier | App FastAPI, rutas de eventos y acceso a la base |
| `web/api/consulta.py` | Tomás | `/api/consulta` (esqueleto que responde 501) |
| `web/requirements.txt` | Javier | `fastapi` y `psycopg[binary]` |
| `contrato/` | Los dos | `esquema.sql`, `eventos_ejemplo.json`, `sembrar_eventos.py`, `CONTRATO.md` |

Más detalle en `PLAN_INTEGRACION.md` (raíz del repo).
