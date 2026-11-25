"""
db.py - Acceso a la tabla `eventos` (dueño: Javier; Tomás lo usa sin modificarlo).

- Si existe la variable DATABASE_URL  -> PostgreSQL (la base que pasa River).
- Si no existe                        -> SQLite local (web/eventos_local.db), para trabajar
                                         hoy sin depender de nadie. En Vercel sin DATABASE_URL
                                         usa /tmp, que se borra: solo sirve para probar.

Funciones públicas (las mismas con las dos bases):
    insertar_evento(evento: dict) -> int
    listar_eventos(desde=None, hasta=None, clase=None, camara=None, limite=50) -> list[dict]
    motor() -> "postgres" | "sqlite"

Las fechas entran y salen como datetime con zona horaria (UTC).
"""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

def _cargar_env():
    """Lee web/.env (si existe) sin pisar variables ya definidas. En Vercel no existe: usa sus variables."""
    ruta = Path(__file__).resolve().parents[1] / ".env"
    if not ruta.exists():
        return
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            clave, valor = linea.split("=", 1)
            os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


_cargar_env()
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
COLUMNAS = ["id", "inicio", "fin", "camara", "clase", "confianza", "confianza_max",
            "n_frames", "modo", "miniatura_url", "creado_en"]
_ESQUEMA_PG = Path(__file__).resolve().parents[2] / "contrato" / "esquema.sql"
_ESQUEMA_SQLITE = """
create table if not exists eventos (
    id integer primary key autoincrement,
    inicio text not null, fin text not null,
    camara text not null default 'webcam_1',
    clase text not null check (clase in ('bovino', 'caballo', 'gato', 'perro')),
    confianza real not null check (confianza between 0 and 1),
    confianza_max real not null check (confianza_max between 0 and 1),
    n_frames integer not null check (n_frames > 0),
    modo text not null check (modo in ('rapido', 'preciso')),
    miniatura_url text,
    creado_en text not null,
    check (fin >= inicio)
);
create index if not exists eventos_inicio_idx on eventos (inicio desc);
"""
_listo = False


def motor():
    return "postgres" if DATABASE_URL else "sqlite"


def _ruta_sqlite():
    if os.environ.get("SQLITE_PATH"):
        return os.environ["SQLITE_PATH"]
    if os.environ.get("VERCEL"):
        return "/tmp/eventos.db"
    return str(Path(__file__).resolve().parents[1] / "eventos_local.db")


@contextmanager
def _conexion():
    global _listo
    if DATABASE_URL:
        import psycopg
        con = psycopg.connect(DATABASE_URL, autocommit=True)
        try:
            if not _listo:
                sql = _ESQUEMA_PG.read_text(encoding="utf-8") if _ESQUEMA_PG.exists() else None
                if sql:
                    con.execute(sql)
                _listo = True
            yield con
        finally:
            con.close()
    else:
        con = sqlite3.connect(_ruta_sqlite())
        con.row_factory = sqlite3.Row
        try:
            if not _listo:
                con.executescript(_ESQUEMA_SQLITE)
                _listo = True
            yield con
            con.commit()
        finally:
            con.close()


def _utc(dt):
    if dt is None:
        return None
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(dt):
    """SQLite guarda texto ISO en UTC con precisión fija: así ordenar y comparar texto = fechas."""
    return _utc(dt).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def insertar_evento(e):
    campos = ["inicio", "fin", "camara", "clase", "confianza", "confianza_max", "n_frames", "modo",
              "miniatura_url"]
    with _conexion() as con:
        if DATABASE_URL:
            valores = [_utc(e["inicio"]), _utc(e["fin"])] + [e.get(c) for c in campos[2:]]
            fila = con.execute(
                f"insert into eventos ({', '.join(campos)}) values ({', '.join(['%s'] * len(campos))}) "
                "returning id", valores).fetchone()
            return int(fila[0])
        valores = [_iso(e["inicio"]), _iso(e["fin"])] + [e.get(c) for c in campos[2:]] + [
            _iso(datetime.now(timezone.utc))]
        cur = con.execute(
            f"insert into eventos ({', '.join(campos)}, creado_en) values ({', '.join(['?'] * (len(campos) + 1))})",
            valores)
        return int(cur.lastrowid)


def listar_eventos(desde=None, hasta=None, clase=None, camara=None, limite=50):
    """Eventos más recientes primero. `desde`/`hasta` filtran por `inicio` (desde <= inicio < hasta)."""
    marca = "%s" if DATABASE_URL else "?"
    convertir = _utc if DATABASE_URL else _iso
    condiciones, valores = [], []
    if desde is not None:
        condiciones.append(f"inicio >= {marca}"); valores.append(convertir(desde))
    if hasta is not None:
        condiciones.append(f"inicio < {marca}"); valores.append(convertir(hasta))
    if clase:
        condiciones.append(f"clase = {marca}"); valores.append(clase)
    if camara:
        condiciones.append(f"camara = {marca}"); valores.append(camara)
    donde = ("where " + " and ".join(condiciones)) if condiciones else ""
    sql = f"select {', '.join(COLUMNAS)} from eventos {donde} order by inicio desc limit {int(limite)}"
    with _conexion() as con:
        filas = con.execute(sql, valores).fetchall()
    eventos = []
    for f in filas:
        d = dict(zip(COLUMNAS, f)) if DATABASE_URL else dict(f)
        for c in ("inicio", "fin", "creado_en"):
            d[c] = _utc(d[c])
        eventos.append(d)
    return eventos
