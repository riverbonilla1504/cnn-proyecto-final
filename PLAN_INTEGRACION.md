# Plan de integración: web con CNN (Javier) + transformer (Tomás)

25 de noviembre de 2026 · Javier

Javier y Tomás trabajan en paralelo sin esperarse: el día 1 congelan un contrato (tabla de eventos, rutas de la API y dueño de cada carpeta) y después cada uno toca solo sus carpetas hasta los dos puntos de integración. Los dos modelos (la CNN y el transformer) corren en el navegador con ONNX; FastAPI en Vercel solo guarda y consulta eventos en la base PostgreSQL de River.

## Arquitectura

```text
Los dos modelos corren en el navegador; solo comparten PostgreSQL

┌─ Navegador · página publicada en Vercel ─────────────────────────────────────┐
│                                                                              │
│  ┌─ Página de cámara · JAVIER ─────────────┐  ┌─ Chat · TOMÁS ─────────────┐ │
│  │ Webcam / video / pestaña                │  │ transformer.onnx entiende  │ │
│  │   → cnn_rapido.onnx o cnn_preciso.onnx  │  │ la pregunta                │ │
│  │ Agrupa frames en eventos y los          │  │ Se monta en #chat con      │ │
│  │ envía a la API                          │  │ montarChat()               │ │
│  └───────────────────┬─────────────────────┘  └─────────────┬──────────────┘ │
└──────────────────────┼──────────────────────────────────────┼────────────────┘
                       │ POST al terminar                     │ intención, clase,
                       │ cada evento                          │ periodo
┌─ Servidor · FastAPI ─┼──────────────────────────────────────┼────────────────┐
│                      ▼                                      ▼                │
│  ┌─ /api/eventos · JAVIER ─────────────────┐  ┌─ /api/consulta · TOMÁS ────┐ │
│  │ POST de un evento · GET del historial   │  │ Consulta la tabla y arma   │ │
│  │                                         │  │ la respuesta               │ │
│  └───────────────────┬─────────────────────┘  └─────────────┬──────────────┘ │
└──────────────────────┼──────────────────────────────────────┼────────────────┘
                       │ guarda                               │ lee
                       ▼                                      ▼
          ┌─ PostgreSQL de River · tabla eventos (COMPARTIDO) ───────────┐
          │      La única unión entre las dos partes (contrato v1)       │
          └──────────────────────────────────────────────────────────────┘
```

La cámara y el chat no se llaman entre sí: la cámara escribe eventos y el chat los lee. Por eso cada parte se construye y se prueba por separado.

## Contrato v1 (se congela el día 1)

El contrato es lo único que los dos comparten. Vive en la carpeta `contrato/` del repo y solo cambia con un PR que aprueban los dos.

### Tabla `eventos` en PostgreSQL

Un registro por aparición de un animal, no uno por cada frame. El esquema está en `contrato/esquema.sql`.

| Columna | Tipo | Regla |
| --- | --- | --- |
| `id` | bigint identity | Lo pone la base |
| `inicio` | timestamptz (UTC) | Momento en que se confirmó el animal |
| `fin` | timestamptz (UTC) | Último frame en que se vio |
| `camara` | text | Según de dónde viene la imagen: `webcam_1` (webcam), `video` (video subido), `pestana` (pestaña compartida) o `simulada` (eventos falsos de prueba) |
| `clase` | text | Exactamente `bovino`, `caballo`, `gato` o `perro` (minúsculas, sin tilde). `fondo` nunca se guarda |
| `confianza` | real (0–1) | Promedio de los frames del evento |
| `confianza_max` | real (0–1) | La más alta del evento |
| `n_frames` | int | Frames clasificados dentro del evento |
| `modo` | text | `rapido` o `preciso` |
| `miniatura_url` | text, opcional | Se deja vacía en la v1 |
| `creado_en` | timestamptz | `default now()` |

### Reglas

- **Zona horaria:** la base guarda todo en UTC. "Hoy", "ayer" y "esta semana" se calculan en hora de Colombia (America/Bogota, UTC−5). Esa conversión la hace la consulta de Tomás.
- **Cuándo hay un evento:** empieza cuando la misma clase sale como animal (no fondo y con confianza ≥ umbral) en 3 frames seguidos. Termina cuando pasan 2 s sin verla. Se envía al terminar.
- **Seguridad:** el navegador nunca habla directo con la base; solo FastAPI, con la cadena de conexión en la variable `DATABASE_URL`, guardada en Vercel. Si `DATABASE_URL` no existe, la API usa SQLite local (`web/eventos_local.db`) automáticamente.

### Rutas de la API (FastAPI, prefijo `/api`)

| Ruta | Dueño | Qué hace |
| --- | --- | --- |
| `GET /api/salud` | Javier | Responde `{"ok": true}` para probar el despliegue |
| `POST /api/eventos` | Javier | Recibe un evento sin `id` ni `creado_en`; devuelve `{"id": 123}` |
| `GET /api/eventos` | Javier | Filtros opcionales `desde`, `hasta` (ISO 8601), `clase`, `camara`, `limite` (50 por defecto); más reciente primero |
| `POST /api/consulta` | Tomás | Recibe lo que entendió el transformer y devuelve la respuesta del chat. El formato interno lo define Tomás, porque solo lo usa su propio chat |

### Cómo se monta el chat en la página

- La página de Javier tiene un contenedor `<section id="chat">` y carga el módulo `/chat/chat.js`.
- Tomás exporta `montarChat(contenedor, { apiBase: "/api" })` y no toca nada fuera de ese contenedor.
- Las clases CSS de Tomás empiezan por `chat-` y las de Javier por `cam-`. Así los estilos no se pisan.
- onnxruntime-web se carga una sola vez desde la página de Javier, con una versión fija que se acuerda el día 1. El chat usa ese mismo `ort`.

### Dependencias

- `web/requirements.txt` lleva FastAPI y `psycopg[binary]` (cliente de PostgreSQL). Nada de TensorFlow: Vercel limita las funciones de Python a 250 MB y TensorFlow no cabe.
- El entrenamiento de Tomás usa su propio entorno, fuera de `web/`.

## Quién es dueño de cada carpeta

Cada archivo tiene un solo dueño. Si alguien necesita cambiar algo que no es suyo, lo pide en un PR y lo aprueba el dueño.

| Ruta | Dueño | Nota |
| --- | --- | --- |
| `src/`, `modelos/`, `docs/`, `historial/` | River | No se tocan. Javier solo agrega dos scripts nuevos en `src/`: `src/exportar_onnx.py` y `src/verificar_onnx.py` (ya existen) |
| `contrato/` | Los dos | `esquema.sql`, `eventos_ejemplo.json`, `sembrar_eventos.py` y `CONTRATO.md`. Cambia solo con un PR que aprueban los dos |
| `web/api/index.py`, `eventos.py`, `db.py` | Javier | `index.py` es la app FastAPI (en local sirve también `web/public`) y registra el router de `consulta.py`. `db.py` da `insertar_evento`, `listar_eventos` y `motor` (conexión a la base), que también usa Tomás sin modificarlo |
| `web/api/consulta.py` | Tomás | Javier lo crea el día 1 como esqueleto que responde 501; desde ahí es de Tomás |
| `web/public/` (menos `chat/`) | Javier | Página de cámara (`index.html`, `js/modelo.js`, `js/eventos.js`, `js/app.js`), estilos `css/estilos.css` (clases `cam-`) y `modelos/` (`cnn_rapido.onnx`, `cnn_preciso.onnx`, `modelo.json`, `esperado.json`) |
| `web/public/chat/` | Tomás | `chat.js`, `transformer.onnx`, `vocab.json` y `demo.html` para probar el chat por separado |
| `transformer/` | Tomás | Generador de preguntas, entrenamiento y exportación |
| `web/requirements.txt`, `vercel.json` | Javier | Tomás pide por PR si necesita otra librería |

Con este reparto, los dos solo comparten la carpeta `contrato/`, y las funciones de `index.py` y `db.py` que Javier deja listas el día 1.

## Exportar a ONNX sin perder precisión

La precisión se conserva exportando en float32, sin cuantizar, y metiendo dentro del archivo ONNX todo lo que hoy hace Python. Después se demuestra con una prueba de paridad. Los `.keras` originales no se tocan: los ONNX son archivos derivados que un script vuelve a generar cuando haga falta.

1. **float32, sin cuantizar.** Pasar a int8 o float16 achica el archivo, pero cambia los números. Con 5–15 MB no hace falta.
2. **Dos archivos:**
   - `cnn_rapido.onnx` (~4.8 MB): solo la red de semilla 42, igual que `--rapido` en Python.
   - `cnn_preciso.onnx` (~14.5 MB): las 3 redes, la foto normal y en espejo y el promedio, todo dentro de un solo grafo. La web hace una sola llamada y el promedio es el mismo cálculo de Python, sin lógica en JavaScript que pueda fallar.
3. **El redimensionado también va dentro del ONNX.** Se agrega una capa de redimensionado bilineal, igual a la de `clasificador.py`. La web pasa el frame RGB a su tamaño real (por ejemplo, 640×480) y el modelo hace el resto. Así el canvas del navegador no redimensiona nada, que es la principal fuente de diferencias.
4. **Entorno:** un `.venv` con Python 3.12, como pide el README (tu PC tiene 3.13).
5. **Criterio de terminado (paridad):**
   - `src/verificar_onnx.py` compara Keras y ONNX con las 10 fotos de `fotos_prueba/`: la misma clase y menos de 0.0001 de diferencia en cada probabilidad.
   - Si se baja `dataset_cnn_maximo.zip` del Drive (1.2 GB), el ONNX preciso debe acertar las mismas 476 de 500 fotos del test (95.2 %). Esto sirve como prueba en la sustentación.
   - Una página `web/public/prueba_onnx.html` corre las mismas 10 fotos en el navegador y las compara con un `esperado.json` que genera Python. Aquí la tolerancia es 0.001, porque el navegador decodifica los JPG un poco distinto.

## Plan de Javier

Javier entrega la web con la cámara, la CNN en ONNX y la API de eventos. Cada fase termina con algo que se puede probar.

| Fase | Qué se hace | Terminado cuando |
| --- | --- | --- |
| J0 — Día 1 (con Tomás) | Base PostgreSQL de River y tabla `eventos` (`contrato/esquema.sql`); carpetas y esqueletos (`index.py`, `db.py`, `consulta.py` con 501, `chat/` vacía); `contrato/` con unos 20 eventos de ejemplo hechos a mano; primer despliegue en Vercel con *Root Directory* = `web/` | `GET /api/salud` responde en la URL de Vercel y el contrato está en `main` |
| J1 — ONNX | `exportar_onnx.py` y `verificar_onnx.py` (sección anterior) | Las pruebas de paridad pasan |
| J2 — API de eventos | `POST` y `GET /api/eventos` con validación (clase permitida, confianza entre 0 y 1, `fin` ≥ `inicio`); script `contrato/sembrar_eventos.py`, que carga ~300 eventos falsos de los últimos 14 días con `camara = simulada` | Los eventos falsos están en la base y se leen con `GET` desde Vercel. **Esto desbloquea la fase T4 de Tomás** |
| J3 — Página de cámara y video | Selector de fuente con tres opciones: webcam, subir un video (.mp4) o compartir una pestaña (por ejemplo, con una cámara en vivo pública). Las tres pasan por el mismo código: onnxruntime-web, modo rápido o preciso, umbral (0.50–0.90, 0.60 por defecto), barras de las 5 probabilidades, latencia y fps. También permite subir una foto (modo preciso). En modo rápido clasifica 3–5 frames por segundo | Las tres fuentes funcionan en local y en Vercel (HTTPS), y hay 2–3 videos de prueba elegidos para la demo |
| J4 — Eventos en vivo | Agrupar frames en eventos con las reglas del contrato, `POST` al terminar cada evento y un panel de historial con `GET` | Un animal mostrado unos 5 s genera un solo evento en la base |
| J5 — Integración final | Montar el chat de Tomás en `#chat` y hacer la prueba de punta a punta (sección Cronograma) | La prueba de punta a punta pasa en Vercel |
| Extra, si sobra tiempo | Recortar la zona con movimiento (diferencia entre frames) antes de clasificar, porque la CNN se entrenó con recortes | Un animal pequeño en la escena se clasifica mejor que con el frame completo |

**Videos para la demo.** En el aula no va a haber animales, así que la demo se basa en videos ya probados. La webcam y la pestaña en vivo quedan como extra.

- **De dónde sacarlos:** videos gratuitos de Pexels o Pixabay (búsquedas como *cows crossing road*, *horse on road*, *stray dog street*) o grabados con el celular. Los propios convencen más, porque el modelo nunca los vio. No se descargan videos de YouTube: sus condiciones no lo permiten.
- **Qué videos elegir:** donde el animal se vea grande, porque la CNN se entrenó con fotos donde el animal llena la imagen. Si el animal está lejos, hace falta la fase extra (recortar la zona con movimiento).
- **Cámaras en vivo:** la página no puede leer un video de YouTube incrustado, porque el navegador lo bloquea. Se usan abriéndolas en otra pestaña y compartiéndola como fuente.
- **Limitación conocida:** el modelo no aprendió qué es una persona, así que en la webcam puede marcar a alguien como animal. La regla de 3 frames seguidos reduce esas falsas alarmas, pero no las elimina. Conviene decirlo en la sustentación.

## Plan de Tomás

Tomás entrega un transformer propio, entrenado desde cero, que entiende la pregunta. La respuesta se arma con datos reales de la tabla `eventos`, así que no puede inventar números. Hasta la fase T4 no necesita nada de Javier.

| Fase | Qué se hace | Terminado cuando |
| --- | --- | --- |
| T0 — Día 1 (con Javier) | Acordar el contrato | El contrato está en `main` |
| T1 — Datos | `transformer/generar_dataset.py`: plantillas, sinónimos (vaca, res, ganado…), errores de tipeo y preguntas fuera de tema. Unas 7 intenciones (contar, listar, último avistamiento, resumen, hora más frecuente, comparar, fuera de dominio), con clase (las 4 o "animales") y periodo (hoy, ayer, última hora, esta semana). Aparte, el grupo escribe a mano 100–200 preguntas reales, que hacen de test | Hay miles de ejemplos generados y el test de preguntas reales está separado |
| T2 — Modelo | Tokenizador propio (vocabulario sacado de los datos, sin tildes ni mayúsculas). Encoder transformer en Keras: 2–4 bloques de atención y 3 salidas (intención, clase, periodo), de 0.5–1 M de parámetros | Supera la meta acordada en las preguntas reales (ver Preguntas abiertas) |
| T3 — ONNX en el navegador | Exportar a `transformer.onnx` y escribir en JavaScript el mismo tokenizador, usando `vocab.json` | Las preguntas reales dan la misma predicción en Python y en el navegador |
| T4 — Consulta | `web/api/consulta.py`: de la intención a la consulta en PostgreSQL (usando `db.py`), con fechas en hora de Colombia y respuesta con plantilla. Antes de J2 prueba en local con `eventos_ejemplo.json`; después, con los eventos `simulada` | "¿Cuántas vacas pasaron hoy?" da el mismo número que contar a mano en la base |
| T5 — Chat | `chat.js` con `montarChat()` y `demo.html`, que funciona sin la página de cámara. Si la confianza es baja, responde "No entendí" con ejemplos de preguntas | `demo.html` funciona en la URL de Vercel de su rama |
| T6 — Integración final | Junto con J5 | La prueba de punta a punta pasa |

## Cronograma

```text
Los dos trabajan en paralelo y solo se sincronizan en tres puntos

┌──────────────────────────────────────────────────────────────────────────────┐
│ P0 · Día 1, juntos: contrato v1 en main (tabla, rutas, carpetas, esqueletos) │
└──────────────────────────────────────────────────────────────────────────────┘

   JAVIER · web, CNN y API de eventos       TOMÁS · transformer y chat

   ┌────────────────────────────────┐       ┌────────────────────────────────┐
   │ J1 · Exportar a ONNX           │       │ T1 · Dataset de preguntas      │
   │ Paridad con Python (10 fotos)  │       │ Generadas + 100–200 reales     │
   └────────────────────────────────┘       └────────────────────────────────┘
   ┌────────────────────────────────┐       ┌────────────────────────────────┐
   │ J2 · API de eventos            │──┐    │ T2 · Transformer en Keras      │
   │ Eventos falsos en PostgreSQL   │  │    │ Intención, clase y periodo     │
   └────────────────────────────────┘  │    └────────────────────────────────┘
   ┌────────────────────────────────┐  │    ┌────────────────────────────────┐
   │ J3 · Página de cámara y video  │ ◇P1   │ T3 · ONNX y tokenizador en JS  │
   │ Webcam, video o pestaña        │  │    │ Misma predicción que Python    │
   └────────────────────────────────┘  │    └────────────────────────────────┘
   ┌────────────────────────────────┐  │    ┌────────────────────────────────┐
   │ J4 · Eventos en vivo           │  └───▶│ T4 · /api/consulta             │
   │ Un evento por aparición        │       │ Usa los eventos que deja J2    │
   └───────────────┬────────────────┘       └────────────────────────────────┘
                   │                        ┌────────────────────────────────┐
                   │                        │ T5 · Chat y demo.html          │
                   │                        │ montarChat() por separado      │
                   │                        └───────────────┬────────────────┘
                   ▼                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ P2 · Integración final (J5 + T6): chat en la página y prueba de punta a punta│
└──────────────────────────────────────────────────────────────────────────────┘
```

P1 es la única espera: Tomás necesita la API de eventos (J2) para probar `/api/consulta` contra la base; mientras tanto usa `eventos_ejemplo.json`. **La entrega es hoy, 25/11/2026**, así que todo el plan se hace hoy.

**Prueba de punta a punta (P2):** se muestra un perro a la webcam unos 5 s, se ve el evento en el historial, se pregunta en el chat "¿qué animales pasaron hoy?" y la respuesta incluye ese perro con su hora.

## Reglas de Git para no chocar

1. River les da acceso de colaborador en `riverbonilla1504/cnn-proyecto-final` a los dos.
2. Cada uno trabaja en su propia rama (`javier/...`, `tomas/...`) y entra a `main` solo por PR.
3. Cada PR toca solo las carpetas de su autor (sección de dueños). Si toca `contrato/`, lo aprueba el otro.
4. Antes de abrir un PR, cada uno actualiza su rama con lo último de `main`.
5. Vercel crea una URL de prueba por rama: cada uno prueba ahí sin tocar la del otro ni la principal.
6. Las claves nunca van al repo: se agregan `.env`, `web/.vercel/` y `web/eventos_local.db` al `.gitignore`, y `DATABASE_URL` vive solo en Vercel y en el `.env` local de cada uno.
7. Los `.onnx` (5–15 MB) sí se suben, porque la web los necesita. Están lejos del límite de 100 MB de GitHub.

## Decisiones tomadas

- [x] ¿El profesor acepta onnxruntime-web, FastAPI y la base? Sí, resuelta.
- [x] Fecha de entrega: hoy, 25/11/2026.
- [x] Regla de alerta: eliminada (no hay columna `alerta` ni aviso en la UI).
- [x] Acierto mínimo del transformer en las preguntas reales: lo define Tomás.
- [ ] Base de datos: PostgreSQL de River (pendiente que la pase y que dé la `DATABASE_URL`).
