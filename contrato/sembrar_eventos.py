"""
sembrar_eventos.py - Genera eventos falsos (camara="simulada") y los envía a la API.

Sirve para llenar la base con datos de prueba y que el asistente (Tomás) y el panel
tengan algo que consultar. Solo usa la librería estándar de Python.

Uso desde PowerShell, en la raíz del repo:

    .venv\\Scripts\\python.exe contrato\\sembrar_eventos.py --url http://localhost:8000

Opciones:
    --url URL            base de la API (por defecto http://localhost:8000; en Vercel: https://tu-app.vercel.app)
    -n, --cantidad N     cuántos eventos generar (por defecto 300)
    -d, --dias D         repartidos en los últimos D días (por defecto 14)
    --semilla S          semilla del azar, para repetir exactamente los mismos eventos
    --solo-json ARCHIVO  escribe los eventos en ese archivo JSON y NO envía nada

Ejemplos:
    .venv\\Scripts\\python.exe contrato\\sembrar_eventos.py --solo-json eventos_falsos.json --semilla 1
    .venv\\Scripts\\python.exe contrato\\sembrar_eventos.py --url https://mi-app.vercel.app -n 100 -d 7

Reglas del contrato (contrato/esquema.sql): una fila por aparición, fechas UTC con zona,
confianza (promedio) <= confianza_max, n_frames >= 1, fin >= inicio.
"""
import argparse
import json
import random
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

BOGOTA = timezone(timedelta(hours=-5))  # Colombia no usa horario de verano

CLASES = ["perro", "gato", "bovino", "caballo"]
PESOS_CLASE = [38, 32, 15, 15]  # perros y gatos más frecuentes

# Peso relativo de cada hora local (0-23): más actividad de día que de noche.
PESOS_HORA = [1, 1, 1, 1, 1, 2, 5, 8, 9, 9, 8, 7, 7, 7, 8, 8, 8, 7, 5, 4, 3, 2, 2, 1]


def generar_evento(rng, ahora, dias):
    desfase_dia = rng.randint(0, dias - 1)
    hora = rng.choices(range(24), weights=PESOS_HORA)[0]
    local = (ahora.astimezone(BOGOTA) - timedelta(days=desfase_dia)).replace(
        hour=hora, minute=rng.randint(0, 59), second=rng.randint(0, 59), microsecond=0)
    inicio = local.astimezone(timezone.utc)

    duracion = min(30, max(1, int(rng.lognormvariate(1.9, 0.7))))  # 1-30 s, mediana ~7 s
    # Nada en el futuro ni fuera de la ventana de D días.
    tope = ahora - timedelta(seconds=duracion + 1)
    if inicio > tope:
        inicio = tope - timedelta(seconds=rng.randint(0, 3600))
    inicio = max(inicio, ahora - timedelta(days=dias))
    fin = inicio + timedelta(seconds=duracion)

    modo = rng.choices(["rapido", "preciso"], weights=[70, 30])[0]
    fps = 4 if modo == "rapido" else 1
    n_frames = max(1, round(duracion * fps * rng.uniform(0.85, 1.05)))

    confianza_max = round(rng.uniform(0.65, 0.99), 4)
    confianza = min(round(max(0.6, confianza_max - rng.uniform(0, 0.12)), 4), confianza_max)

    return {
        "inicio": inicio.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fin": fin.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "camara": "simulada",
        "clase": rng.choices(CLASES, weights=PESOS_CLASE)[0],
        "confianza": confianza,
        "confianza_max": confianza_max,
        "n_frames": n_frames,
        "modo": modo,
    }


def enviar(url, evento):
    req = urllib.request.Request(
        url.rstrip("/") + "/api/eventos",
        data=json.dumps(evento).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))["id"]


def main():
    p = argparse.ArgumentParser(description="Siembra eventos falsos en la API de eventos.")
    p.add_argument("--url", default="http://localhost:8000", help="base de la API")
    p.add_argument("-n", "--cantidad", type=int, default=300, help="número de eventos")
    p.add_argument("-d", "--dias", type=int, default=14, help="últimos D días")
    p.add_argument("--semilla", type=int, default=None, help="semilla para reproducibilidad")
    p.add_argument("--solo-json", metavar="ARCHIVO", help="escribir a un archivo en vez de enviar")
    a = p.parse_args()
    if a.cantidad < 1 or a.dias < 1:
        p.error("--cantidad y --dias deben ser >= 1")

    rng = random.Random(a.semilla)
    ahora = datetime.now(timezone.utc).replace(microsecond=0)
    eventos = sorted((generar_evento(rng, ahora, a.dias) for _ in range(a.cantidad)),
                     key=lambda e: e["inicio"])

    if a.solo_json:
        with open(a.solo_json, "w", encoding="utf-8") as f:
            json.dump(eventos, f, ensure_ascii=False, indent=2)
        print(f"{len(eventos)} eventos escritos en {a.solo_json} (no se envió nada).")
        return 0

    guardados, errores = 0, 0
    for e in eventos:
        try:
            enviar(a.url, e)
            guardados += 1
        except urllib.error.HTTPError as ex:
            errores += 1
            detalle = ex.read().decode("utf-8", "replace")[:200]
            print(f"Error {ex.code} al guardar {e['clase']} {e['inicio']}: {detalle}", file=sys.stderr)
        except (urllib.error.URLError, OSError) as ex:
            errores += 1
            print(f"No se pudo conectar con {a.url}: {ex}", file=sys.stderr)
            print("Se detiene: ¿está corriendo el servidor?", file=sys.stderr)
            break
    print(f"Guardados: {guardados} de {len(eventos)}. Errores: {errores}.")
    return 0 if errores == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
