"""
PASO 1 - Ampliar el dataset con MÁS recortes de Open Images (misma fuente, mismas reglas).

Por qué: en los experimentos previos (historial/) la CNN desde cero se quedó en ~69 % y sus curvas muestran que el límite es la
CANTIDAD de datos (2 125 fotos de entrenamiento). Open Images tiene miles de cajas más de
nuestras clases; el CSV de anotaciones ya está en el proyecto.

Reglas (las mismas con que se armó data/dataset_cnn, para que todo sea comparable):
  - Cajas de UN animal (IsGroupOf = 0), fotos reales, no dibujos (IsDepiction = 0),
    etiqueta verificada por personas (Confidence = 1).
  - Máximo un recorte por clase por foto original.
  - "fondo" = cajas de autos y camiones en fotos SIN ninguna de las 4 especies.
  - NO se repite ninguna foto ya usada en dataset_cnn/train.
  - El TEST (dataset_cnn/val, partición oficial "validation" de Open Images) NO se toca:
    las fotos nuevas salen todas de la partición "train" de Open Images -> no hay fuga.
  - Recorte guardado con lado mayor <= 384 px (igual que el dataset original);
    se descartan recortes con algún lado < 32 px (no se ve nada).

Qué hace:
  1. Elige cajas candidatas al azar (semilla fija) en las anotaciones.
  2. Descarga cada foto original del repositorio público de Open Images (en paralelo),
     recorta la caja y guarda SOLO el recorte (la foto completa no se guarda).
  3. Arma data/dataset_cnn_ampliado/ = las 500 originales + las nuevas por clase,
     con el mismo test de 500. El dataset original no se modifica.
  4. manifest_train.csv con el origen de cada recorte ("original" o "nuevo") y su orden,
     para poder construir subconjuntos anidados (se usó en la curva de aprendizaje, historial/).

Uso:  .venv\\Scripts\\python.exe src\\paso1_ampliar_dataset.py --nuevas 1800
"""
import argparse
import io
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from PIL import Image

from comun import CLASES, RAIZ

ORIGEN = RAIZ / "data" / "dataset_cnn"
DESTINO = RAIZ / "data" / "dataset_cnn_ampliado"
DIR_ANOT = RAIZ / "data" / "anotaciones_open_images"
CSV_COMPLETO = DIR_ANOT / "oidv6-train-annotations-bbox.csv"   # todas las cajas de Open Images train
URL = "https://open-images-dataset.s3.amazonaws.com/train/{}.jpg"
LADO_MAX = 384
LADO_MIN = 32
SEED = 42
ETIQUETAS_FONDO = {"/m/0k4j": "Car", "/m/07r04": "Truck"}

parser = argparse.ArgumentParser()
parser.add_argument("--nuevas", type=int, default=1800, help="recortes nuevos por clase")
parser.add_argument("--hilos", type=int, default=24, help="descargas en paralelo")
args = parser.parse_args()

filtro = lambda d: d[(d.IsGroupOf == 0) & (d.IsDepiction == 0) & (d.Confidence == 1)]

# 1. Candidatos ----------------------------------------------------------------------
manifest_orig = pd.read_csv(ORIGEN / "manifest_train.csv")
usadas = set(manifest_orig.image_id)

animales = pd.read_csv(DIR_ANOT / "open_images_animales_train.csv")
con_animal = set(animales.ImageID)          # fotos que tienen alguna de las 4 especies
cand = filtro(animales)
cand = cand[~cand.ImageID.isin(usadas)].rename(columns={"Clase": "clase"})

print("Buscando autos y camiones en el CSV completo de Open Images (2.2 GB, tarda unos minutos)...")
trozos = []
for t in pd.read_csv(CSV_COMPLETO, chunksize=2_000_000,
                     usecols=["ImageID", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax",
                              "IsGroupOf", "IsDepiction"]):
    trozos.append(t[t.LabelName.isin(ETIQUETAS_FONDO)])
fondo = filtro(pd.concat(trozos))
fondo = fondo[~fondo.ImageID.isin(con_animal) & ~fondo.ImageID.isin(usadas)].assign(clase="fondo")
cand = pd.concat([cand, fondo], ignore_index=True)

# un recorte por clase por foto (caja elegida al azar), luego orden aleatorio por clase
cand = (cand.sample(frac=1, random_state=SEED)
            .drop_duplicates(["clase", "ImageID"])
            .reset_index(drop=True))
print("Fotos candidatas por clase:", cand.groupby("clase").size().to_dict())

# 2. Descarga y recorte ----------------------------------------------------------------
DIR_NUEVAS = DESTINO / "_nuevas"     # recortes nuevos (se conservan si se vuelve a correr)


def recortar(fila):
    salida = DIR_NUEVAS / fila.clase / f"{fila.ImageID}.jpg"
    if salida.exists():
        return fila.Index, "ok"
    for intento in range(3):
        try:
            datos = urllib.request.urlopen(URL.format(fila.ImageID), timeout=30).read()
            break
        except Exception:
            time.sleep(1 + intento)
    else:
        return fila.Index, "descarga_fallida"
    img = Image.open(io.BytesIO(datos)).convert("RGB")
    w, h = img.size
    caja = (round(fila.XMin * w), round(fila.YMin * h), round(fila.XMax * w), round(fila.YMax * h))
    rec = img.crop(caja)
    if min(rec.size) < LADO_MIN:
        return fila.Index, "muy_pequeno"
    rec.thumbnail((LADO_MAX, LADO_MAX), Image.BILINEAR)
    salida.parent.mkdir(parents=True, exist_ok=True)
    rec.save(salida, quality=92)
    return fila.Index, "ok"


estado = {}
t0 = time.time()
for clase in CLASES:
    pendientes = cand[cand.clase == clase]
    ok = 0
    pos = 0
    # se procesa por tandas hasta juntar args.nuevas recortes válidos
    with ThreadPoolExecutor(args.hilos) as pool:
        while ok < args.nuevas and pos < len(pendientes):
            tanda = pendientes.iloc[pos: pos + (args.nuevas - ok) + 50]
            pos += len(tanda)
            for fut in as_completed([pool.submit(recortar, f) for f in tanda.itertuples()]):
                i, e = fut.result()
                estado[i] = e
                ok += e == "ok"
    malos = pd.Series([estado[i] for i in pendientes.index[:pos]]).value_counts().to_dict()
    print(f"{clase:8s}: {ok} recortes nuevos  {malos}  ({(time.time() - t0) / 60:.1f} min)")

# 3. Armar el dataset ampliado -----------------------------------------------------------
cand["estado"] = cand.index.map(estado)
nuevas = cand[cand.estado == "ok"].copy()
# si una clase consiguió más de las pedidas, se recorta (orden aleatorio ya fijado)
nuevas = nuevas.groupby("clase", group_keys=False).head(args.nuevas)

for sub in ("train", "val"):
    shutil.rmtree(DESTINO / sub, ignore_errors=True)
shutil.copytree(ORIGEN / "val", DESTINO / "val")              # test idéntico
filas = []
for _, r in manifest_orig.iterrows():
    dst = DESTINO / "train" / r.clase / f"{r.image_id}.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ORIGEN / r.archivo, dst)
    filas.append(dict(clase=r.clase, image_id=r.image_id, origen="original",
                      xmin=r.xmin, ymin=r.ymin, xmax=r.xmax, ymax=r.ymax))
for r in nuevas.itertuples():
    dst = DESTINO / "train" / r.clase / f"{r.ImageID}.jpg"
    shutil.copy2(DIR_NUEVAS / r.clase / f"{r.ImageID}.jpg", dst)
    filas.append(dict(clase=r.clase, image_id=r.ImageID, origen="nuevo",
                      xmin=r.XMin, ymin=r.YMin, xmax=r.XMax, ymax=r.YMax))

man = pd.DataFrame(filas)
# orden: primero las 500 originales de cada clase, luego las nuevas en orden aleatorio
man["orden"] = man.groupby("clase").cumcount()
man["archivo"] = "train/" + man.clase + "/" + man.image_id + ".jpg"
man.to_csv(DESTINO / "manifest_train.csv", index=False)
shutil.copy2(ORIGEN / "manifest_val.csv", DESTINO / "manifest_val.csv")

print(f"\nDataset ampliado en {DESTINO}")
print("train por clase:", man.groupby("clase").size().to_dict())
print(f"Tiempo total: {(time.time() - t0) / 60:.1f} min")
