"""
PASO 2 - Ampliar el dataset AL MÁXIMO que permite Open Images.

Parte del dataset ampliado (paso 1, ~2 300 por clase) y agrega más recortes:
  - De la partición "train" de Open Images (las fotos que el paso 1 no alcanzó a usar)
    y de la partición "test" de Open Images (otras ~125 000 fotos con cajas).
  - La partición "validation" NO se usa: de ahí sale nuestro test de 500 fotos.
  - bovino = "Cattle" + "Bull" (toro). Es la clase más escasa y la que pone el límite.
  - Mismas reglas que siempre: IsGroupOf = 0, IsDepiction = 0, Confidence = 1,
    un recorte por clase por foto, lado mínimo 32 px, lado mayor <= 384 px.
  - Tope por clase (--max-por-clase, 6 000 por defecto) para que el entrenamiento no
    tarde demasiado y las clases no queden muy desbalanceadas. Bovino y caballo no
    alcanzan el tope: se usan TODAS sus fotos disponibles.

Salida: data/dataset_cnn_maximo/ (train ampliado + nuevas; val = el mismo test de 500)
        data/dataset_cnn_maximo/manifest_train.csv

Uso:  .venv\\Scripts\\python.exe src\\paso2_ampliar_al_maximo.py --max-por-clase 6000
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

BASE = RAIZ / "data" / "dataset_cnn_ampliado"
DESTINO = RAIZ / "data" / "dataset_cnn_maximo"
DIR_ANOT = RAIZ / "data" / "anotaciones_open_images"
CSV_TRAIN = DIR_ANOT / "oidv6-train-annotations-bbox.csv"
CSV_TEST = DIR_ANOT / "test-annotations-bbox.csv"   # storage.googleapis.com/openimages/v5/
URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg"
LADO_MAX, LADO_MIN, SEED = 384, 32, 42
ETIQUETAS = {"/m/01xq0k1": "bovino", "/m/0cnyhnx": "bovino",   # Cattle, Bull
             "/m/03k3r": "caballo", "/m/01yrx": "gato", "/m/0bt9lr": "perro",
             "/m/0k4j": "fondo", "/m/07r04": "fondo"}             # Car, Truck
ANIMALES = {k for k, v in ETIQUETAS.items() if v != "fondo"}
COLS = ["ImageID", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsGroupOf", "IsDepiction"]

parser = argparse.ArgumentParser()
parser.add_argument("--max-por-clase", type=int, default=6000)
parser.add_argument("--hilos", type=int, default=24)
args = parser.parse_args()

man_base = pd.read_csv(BASE / "manifest_train.csv")
usadas = set(man_base.image_id)


def candidatos(df, split):
    con_animal = set(df[df.LabelName.isin(ANIMALES)].ImageID)
    d = df[df.LabelName.isin(ETIQUETAS) & (df.IsGroupOf == 0) & (df.IsDepiction == 0)
           & (df.Confidence == 1) & ~df.ImageID.isin(usadas)].copy()
    d["clase"] = d.LabelName.map(ETIQUETAS)
    d = d[(d.clase != "fondo") | ~d.ImageID.isin(con_animal)]   # fondo: fotos sin animales
    d["split"] = split
    return d


print("Leyendo anotaciones (el CSV de train pesa 2.2 GB, tarda unos minutos)...")
tr = pd.concat(t[t.LabelName.isin(ETIQUETAS)]
               for t in pd.read_csv(CSV_TRAIN, chunksize=3_000_000, usecols=COLS))
te = pd.read_csv(CSV_TEST, usecols=COLS)
cand = pd.concat([candidatos(tr, "train"), candidatos(te, "test")], ignore_index=True)
cand = (cand.sample(frac=1, random_state=SEED)
            .drop_duplicates(["clase", "ImageID"])
            .reset_index(drop=True))
print("Fotos candidatas nuevas por clase:", cand.groupby("clase").size().to_dict())

DIR_NUEVAS = DESTINO / "_nuevas"


def recortar(fila):
    salida = DIR_NUEVAS / fila.clase / f"{fila.ImageID}.jpg"
    if salida.exists():
        return fila.Index, "ok"
    for intento in range(3):
        try:
            datos = urllib.request.urlopen(URL.format(split=fila.split, id=fila.ImageID), timeout=30).read()
            break
        except Exception:
            time.sleep(1 + intento)
    else:
        return fila.Index, "descarga_fallida"
    try:
        img = Image.open(io.BytesIO(datos)).convert("RGB")
    except Exception:
        return fila.Index, "imagen_danada"
    w, h = img.size
    rec = img.crop((round(fila.XMin * w), round(fila.YMin * h), round(fila.XMax * w), round(fila.YMax * h)))
    if min(rec.size) < LADO_MIN:
        return fila.Index, "muy_pequeno"
    rec.thumbnail((LADO_MAX, LADO_MAX), Image.BILINEAR)
    salida.parent.mkdir(parents=True, exist_ok=True)
    rec.save(salida, quality=92)
    return fila.Index, "ok"


estado, t0 = {}, time.time()
ya_tiene = man_base.groupby("clase").size().to_dict()
for clase in CLASES:
    faltan = max(0, args.max_por_clase - ya_tiene[clase])
    pendientes = cand[cand.clase == clase]
    ok = pos = 0
    with ThreadPoolExecutor(args.hilos) as pool:
        while ok < faltan and pos < len(pendientes):
            tanda = pendientes.iloc[pos: pos + (faltan - ok) + 50]
            pos += len(tanda)
            for fut in as_completed([pool.submit(recortar, f) for f in tanda.itertuples()]):
                i, e = fut.result()
                estado[i] = e
                ok += e == "ok"
    malos = pd.Series([estado[i] for i in pendientes.index[:pos]]).value_counts().to_dict() if pos else {}
    print(f"{clase:8s}: +{ok} nuevas (pedidas {faltan})  {malos}  ({(time.time() - t0) / 60:.1f} min)")

# Armar el dataset: base ampliada + nuevas (en el orden aleatorio fijado) ---------------
cand["estado"] = cand.index.map(estado)
nuevas = cand[cand.estado == "ok"].copy()
nuevas["cupo"] = nuevas.clase.map(lambda c: max(0, args.max_por_clase - ya_tiene[c]))
nuevas = nuevas[nuevas.groupby("clase").cumcount() < nuevas.cupo]

for sub in ("train", "val"):
    shutil.rmtree(DESTINO / sub, ignore_errors=True)
shutil.copytree(BASE / "val", DESTINO / "val")
shutil.copytree(BASE / "train", DESTINO / "train")
filas = man_base[["clase", "image_id", "origen", "xmin", "ymin", "xmax", "ymax"]].to_dict("records")
for r in nuevas.itertuples():
    shutil.copy2(DIR_NUEVAS / r.clase / f"{r.ImageID}.jpg", DESTINO / "train" / r.clase / f"{r.ImageID}.jpg")
    filas.append(dict(clase=r.clase, image_id=r.ImageID, origen=f"nuevo_maximo_{r.split}",
                      xmin=r.XMin, ymin=r.YMin, xmax=r.XMax, ymax=r.YMax))
man = pd.DataFrame(filas)
man["orden"] = man.groupby("clase").cumcount()
man["archivo"] = "train/" + man.clase + "/" + man.image_id + ".jpg"
man.to_csv(DESTINO / "manifest_train.csv", index=False)
shutil.copy2(BASE / "manifest_val.csv", DESTINO / "manifest_val.csv")
print(f"\nDataset en {DESTINO}")
print("train por clase:", man.groupby("clase").size().to_dict(), "| total:", len(man))
print("origen:", man.origen.value_counts().to_dict())
print(f"Tiempo total: {(time.time() - t0) / 60:.1f} min")
