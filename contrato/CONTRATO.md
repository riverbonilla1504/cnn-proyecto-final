# Contrato Javier - Tomás (v1)

Cambios en `contrato/` solo por PR aprobado por los dos. La fuente de verdad del formato es `contrato/esquema.sql`.

## Tabla `eventos`

Una fila por APARICIÓN de un animal (no una por frame). La escribe la página de cámara (Javier) y la lee el asistente (Tomás).

| Columna | Tipo | Regla |
|---|---|---|
| `id` | entero | Lo asigna la base. |
| `inicio` | timestamptz | UTC con zona: cuándo se confirmó el animal. |
| `fin` | timestamptz | UTC con zona: último frame visto. `fin >= inicio`. |
| `camara` | texto | `webcam_1` (defecto), `video`, `pestana` o `simulada`. |
| `clase` | texto | Exactamente `bovino`, `caballo`, `gato` o `perro`. `fondo` nunca se guarda. |
| `confianza` | real 0–1 | Promedio del evento. |
| `confianza_max` | real 0–1 | La más alta del evento; `>= confianza`. |
| `n_frames` | entero | Frames clasificados, `>= 1`. |
| `modo` | texto | `rapido` (~4 frames/s) o `preciso` (~1 frame/s). |
| `miniatura_url` | texto | Opcional; vacía en la v1. |
| `creado_en` | timestamptz | Lo pone la base. |

No existe columna `alerta`. "Hoy" y "ayer" se calculan en hora de Colombia (`America/Bogota`, UTC−5), pero todo se guarda en UTC.

## Rutas de la API

| Ruta | Qué hace |
|---|---|
| `GET /api/salud` | Comprueba que la API vive. |
| `POST /api/eventos` | Guarda un evento (JSON con los campos de arriba, sin `id` ni `creado_en`). Responde 201 `{"id": 123}`; 422 si no cumple las reglas; 503 si la base falla. |
| `GET /api/eventos` | Lista eventos, el más reciente primero. |
| `POST /api/consulta` | Reservada a Tomás (asistente). Hoy responde 501. |

Filtros de `GET /api/eventos` (todos opcionales):

- `desde`, `hasta`: fechas ISO con zona; filtran por `inicio` (`desde <= inicio < hasta`).
- `clase`: una de las 4 clases.
- `camara`: una de las 4 cámaras.
- `limite`: 1–500 (defecto 50).

Ejemplo: `GET /api/eventos?clase=perro&desde=2026-11-25T05:00:00Z&limite=100`.

Datos de prueba: `contrato/eventos_ejemplo.json` (20 eventos, sin base) y `contrato/sembrar_eventos.py` (genera y envía eventos falsos; ver su docstring).

## Leer eventos desde Python

Desde la carpeta `web/` (o con `web/` en el `sys.path`):

```python
from api import db

eventos = db.listar_eventos(clase="gato", limite=100)  # desde, hasta, clase, camara, limite
for e in eventos:
    print(e["clase"], e["inicio"], e["confianza"])
```

Devuelve una lista de dicts con las claves `id, inicio, fin, camara, clase, confianza, confianza_max, n_frames, modo, miniatura_url, creado_en`. `inicio`, `fin` y `creado_en` son `datetime` con zona UTC; conviértalos con `.astimezone(ZoneInfo("America/Bogota"))` para mostrarlos. No modifique `db.py`: es de Javier.

## Cómo se monta el chat

- La página tiene `<section id="chat">`.
- Carga `/chat/chat.js`, que debe exportar `montarChat(contenedor, { apiBase: "/api" })`.
- CSS de Tomás con prefijo `chat-`; el de Javier usa `cam-`. Así no chocan.
- `onnxruntime-web` 1.31.1 lo carga la página una sola vez y queda como global `ort`; el chat no debe cargarlo otra vez.
- El chat habla con la API solo vía `apiBase` (`${apiBase}/consulta`, `${apiBase}/eventos`).

## Dueños de carpetas

| Carpeta | Dueño |
|---|---|
| `transformer/`, `web/public/chat/`, `web/api/consulta.py` | Tomás |
| Resto de `web/` (incluye `web/api/db.py`, `web/api/eventos.py`) | Javier |
| `contrato/` | Ambos: cambios por PR aprobado por los dos |

## Base de datos

- Con la variable de entorno `DATABASE_URL` se usa PostgreSQL, que provee River. La tabla se crea sola desde `contrato/esquema.sql` la primera vez.
- Sin `DATABASE_URL` se usa SQLite local automático (`web/eventos_local.db`), para trabajar sin depender de nadie. En Vercel sin `DATABASE_URL` usa `/tmp`, que se borra: solo sirve para probar.
- Las dos bases tienen las mismas funciones (`insertar_evento`, `listar_eventos`, `motor()`) y devuelven lo mismo.
