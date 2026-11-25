"""
PASO 6 - Dataset v3: animales LEJOS y desde CÁMARAS FIJAS (seguridad / cámara trampa).

Por qué: el modelo v2 (95.2 % en el test de Open Images) falla en video. En una cámara fija
la vaca ocupa una parte pequeña del frame y se ve desde arriba, borrosa y comprimida; v2 solo
vio recortes donde el animal LLENA la foto. Se arregla con DATOS nuevos, no con otra red
preentrenada (sigue prohibido): solo se usan imágenes y cajas.

Fuentes (solo datos; algunas cajas las hizo MegaDetector, un detector preentrenado, y el
grupo decidió que usar sus ETIQUETAS está permitido porque la red se entrena desde cero):
  - Open Images  (las 27 202 fotos del dataset máximo, ahora con CONTEXTO alrededor)
  - COCO 2017    (vaca, caballo, perro, gato; muchas cajas pequeñas; escenas sin animales)
  - ENA24        (cámara trampa, perro/gato/caballo, de día y con IR de noche)
  - Caltech CT   (cámara trampa, perro y gato; miles de escenas VACÍAS = fondo real)
  - WCS          (cámara trampa, 12 países, muchos bovinos)
  - NACTI        (cámara trampa en EE. UU., 2 millones de fotos de ganado; cajas MegaDetector)

Qué genera:
  TRAIN: "recortes con contexto". Alrededor de un animal se recorta un cuadrado k veces más
         grande que su caja (k al azar entre 1.3 y 8). Así el animal ocupa desde casi todo el
         recorte hasta ~2 % de él, como en una cámara lejana. Se guarda a 384x384.
         Fondo: recortes al azar de escenas vacías, calles, personas, autos y otros animales.
  VAL / TEST: FOTOS COMPLETAS (lado mayor 640 px), de UBICACIONES (cámaras) que no aparecen
         en train. Así se mide lo que importa: el frame completo de una cámara nunca vista.
  Etiquetas MULTI-ETIQUETA: un recorte puede tener vaca Y perro. 0 animales = fondo.

Salida: data/dataset_v3/{train,val,test}/<archivos>.jpg + data/dataset_v3/manifest.csv

Uso:  .venv\\Scripts\\python.exe src\\paso6_armar_dataset_v3.py
"""
import argparse
import hashlib
import io
import json
import math
import random
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
META = RAIZ / "data" / "externos" / "metadatos"
ANOT_OI = RAIZ / "data" / "anotaciones_open_images"
DESTINO = RAIZ / "data" / "dataset_v3"
ANIMALES = ["bovino", "caballo", "gato", "perro"]
LADO_TRAIN, LADO_EVAL = 384, 640
SEED = 42

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--hilos", type=int, default=32)
ap.add_argument("--solo-listar", action="store_true", help="solo contar, sin descargar")
args = ap.parse_args()
rng = random.Random(SEED)


def split_por_ubicacion(fuente, ubicacion):
    """10 % de las cámaras a test, 10 % a val, 80 % a train (determinístico)."""
    h = int(hashlib.md5(f"{fuente}|{ubicacion}".encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


registros = []   # dict(fuente, id, url, split, ubic, cajas=[(clase,x0,y0,x1,y1) normalizadas], otras=[...], etiquetas=set)


def agregar(**r):
    r.setdefault("otras", [])
    r["etiquetas"] = sorted({c[0] for c in r["cajas"]} | set(r.get("etiquetas_img", [])))
    r.pop("etiquetas_img", None)
    registros.append(r)


def muestrear(lista, n, clave=None, por_grupo=None):
    """Hasta n elementos al azar; con por_grupo limita cuántos salen de cada cámara."""
    lista = list(lista)
    rng.shuffle(lista)
    if por_grupo:
        cuenta, salida = Counter(), []
        for x in lista:
            if cuenta[clave(x)] < por_grupo:
                cuenta[clave(x)] += 1
                salida.append(x)
        lista = salida
    return lista[:n]


t0 = time.time()
# ---------------------------------------------------------------- 1. Open Images
print("Open Images: cajas de todas nuestras clases en las fotos del dataset máximo...")
ETIQ_OI = {"/m/01xq0k1": "bovino", "/m/0cnyhnx": "bovino", "/m/03k3r": "caballo",
           "/m/01yrx": "gato", "/m/0bt9lr": "perro"}
man = pd.read_csv(RAIZ / "data" / "dataset_cnn_maximo" / "manifest_train.csv")
man["split_oi"] = man.origen.map(lambda o: "test" if o.endswith("_test") else "train")
ids = set(man.image_id)
cols = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"]
cajas_oi = pd.concat(
    [t[t.ImageID.isin(ids) & t.LabelName.isin(ETIQ_OI)]
     for t in pd.read_csv(ANOT_OI / "oidv6-train-annotations-bbox.csv", usecols=cols, chunksize=3_000_000)]
    + [pd.read_csv(ANOT_OI / "test-annotations-bbox.csv", usecols=cols).query("ImageID in @ids and LabelName in @ETIQ_OI")])
cajas_oi["clase"] = cajas_oi.LabelName.map(ETIQ_OI)
por_img = defaultdict(list)
for r in cajas_oi.itertuples():
    por_img[r.ImageID].append((r.clase, r.XMin, r.YMin, r.XMax, r.YMax))
for r in man.drop_duplicates("image_id").itertuples():
    url = f"https://open-images-dataset.s3.amazonaws.com/{r.split_oi}/{r.image_id}.jpg"
    if r.clase == "fondo":        # foto sin animales: el centro del recorte es el auto/camión
        agregar(fuente="openimages", id=r.image_id, url=url, split="train", ubic="oi",
                cajas=[], otras=[("vehiculo", r.xmin, r.ymin, r.xmax, r.ymax)])
    else:
        agregar(fuente="openimages", id=r.image_id, url=url, split="train", ubic="oi",
                cajas=por_img.get(r.image_id) or [(r.clase, r.xmin, r.ymin, r.xmax, r.ymax)])
print(f"  {sum(x['fuente'] == 'openimages' for x in registros)} fotos ({time.time() - t0:.0f} s)")

# ---------------------------------------------------------------- 2. COCO 2017
print("COCO 2017...")
COCO_A = {"cow": "bovino", "horse": "caballo", "cat": "gato", "dog": "perro"}
for parte in ("train2017", "val2017"):
    d = json.load(open(META / "annotations" / f"instances_{parte}.json", encoding="utf-8"))
    cats = {c["id"]: c for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}
    por = defaultdict(list); supers = defaultdict(set)
    for a in d["annotations"]:
        c = cats[a["category_id"]]
        supers[a["image_id"]].add(c["supercategory"])
        if c["name"] in COCO_A:
            im = imgs[a["image_id"]]; x, y, w, h = a["bbox"]
            por[a["image_id"]].append((COCO_A[c["name"]], x / im["width"], y / im["height"],
                                       (x + w) / im["width"], (y + h) / im["height"]))
    if parte == "train2017":
        positivas = list(por)
        otros_animales = [i for i in imgs if i not in por and "animal" in supers[i]]
        sin_animal = [i for i in imgs if "animal" not in supers[i] and supers[i]]
        con_persona = [i for i in sin_animal if "person" in supers[i]]
        sin_persona = [i for i in sin_animal if "person" not in supers[i]]
        negativas = (muestrear(con_persona, 4000) + muestrear(sin_persona, 2500)
                     + muestrear(otros_animales, 1500))
        lista = [(i, "train") for i in positivas + negativas]
    else:     # val2017 = foto completa, mitad val / mitad test
        negs = muestrear([i for i in imgs if i not in por and supers[i]], 600)
        lista = [(i, "val" if i % 2 else "test") for i in list(por) + negs]
    for i, sp in lista:
        im = imgs[i]
        agregar(fuente="coco", id=str(i), url=im["coco_url"], split=sp, ubic="coco", cajas=por.get(i, []))
print(f"  {sum(x['fuente'] == 'coco' for x in registros)} fotos ({time.time() - t0:.0f} s)")

# ---------------------------------------------------------------- 3. ENA24 (todo a train: no trae ubicación)
print("ENA24...")
d = json.load(open(META / "ena24.json", encoding="utf-8"))
cats = {c["id"]: c["name"].lower() for c in d["categories"]}
ENA = {"dog": "perro", "domestic cat": "gato", "horse": "caballo"}
imgs = {i["id"]: i for i in d["images"]}
por = defaultdict(list); otras = defaultdict(list)
for a in d["annotations"]:
    im = imgs[a["image_id"]]; x, y, w, h = a["bbox"]
    caja = (x / im["width"], y / im["height"], (x + w) / im["width"], (y + h) / im["height"])
    (por if cats[a["category_id"]] in ENA else otras)[a["image_id"]].append(
        (ENA.get(cats[a["category_id"]], "otro_animal"),) + caja)
for i in por:
    agregar(fuente="ena24", id=i, url=f"https://storage.googleapis.com/public-datasets-lila/ena24/images/{imgs[i]['file_name']}",
            split="train", ubic="ena24", cajas=por[i], otras=otras.get(i, []))
print(f"  {len(por)} fotos")

# ---------------------------------------------------------------- 4. Caltech Camera Traps
print("Caltech Camera Traps...")
BASE_CCT = "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/"
d = json.load(open(META / "caltech_bboxes_20200316.json", encoding="utf-8"))
cats = {c["id"]: c["name"] for c in d["categories"]}
imgs = {i["id"]: i for i in d["images"]}
por = defaultdict(list)
for a in d["annotations"]:
    if cats[a["category_id"]] in ("cat", "dog") and "bbox" in a:
        im = imgs[a["image_id"]]; x, y, w, h = a["bbox"]
        por[a["image_id"]].append(({"cat": "gato", "dog": "perro"}[cats[a["category_id"]]],
                                   x / im["width"], y / im["height"], (x + w) / im["width"], (y + h) / im["height"]))
for i in por:
    im = imgs[i]
    agregar(fuente="cct", id=i, url=BASE_CCT + im["file_name"], split=split_por_ubicacion("cct", im["location"]),
            ubic=str(im["location"]), cajas=por[i])
d = json.load(open(META / "caltech_images_20210113.json", encoding="utf-8"))
cats = {c["id"]: c["name"] for c in d["categories"]}
imgs = {i["id"]: i for i in d["images"]}
etq = defaultdict(set)
for a in d["annotations"]:
    etq[a["image_id"]].add(cats[a["category_id"]])
vacias = [imgs[i] for i, s in etq.items() if s == {"empty"} and i in imgs]
vacias = muestrear(vacias, 4500, clave=lambda im: im["location"], por_grupo=60)
for im in vacias:
    agregar(fuente="cct", id=im["id"], url=BASE_CCT + im["file_name"], split=split_por_ubicacion("cct", im["location"]),
            ubic=str(im["location"]), cajas=[])
print(f"  {len(por)} con perro/gato, {len(vacias)} vacías ({time.time() - t0:.0f} s)")

# ---------------------------------------------------------------- 5. WCS Camera Traps
print("WCS Camera Traps...")
BASE_WCS = "https://storage.googleapis.com/public-datasets-lila/wcs-unzipped/"
d = json.load(open(META / "wcs_20220205_bboxes_with_classes.json", encoding="utf-8"))
cats = {c["id"]: c["name"].lower() for c in d["categories"]}
WCS = {"bos taurus": "bovino", "canis familiaris": "perro", "equus caballus": "caballo", "horse": "caballo"}
imgs = {i["id"]: i for i in d["images"]}
por = defaultdict(list); vacias = []
for a in d["annotations"]:
    n = cats[a["category_id"]]
    im = imgs[a["image_id"]]
    if n in WCS and "bbox" in a:
        x, y, w, h = a["bbox"]
        por[a["image_id"]].append((WCS[n], x / im["width"], y / im["height"], (x + w) / im["width"], (y + h) / im["height"]))
    elif n == "empty":
        vacias.append(im)
sel_bov = muestrear([i for i in por if por[i][0][0] == "bovino"], 5000, clave=lambda i: imgs[i]["location"], por_grupo=40)
sel = sel_bov + [i for i in por if por[i][0][0] != "bovino"]
for i in sel:
    im = imgs[i]
    agregar(fuente="wcs", id=i, url=BASE_WCS + im["file_name"], split=split_por_ubicacion("wcs", im["location"]),
            ubic=str(im["location"]), cajas=por[i])
vacias = muestrear({im["id"]: im for im in vacias if not im["file_name"].startswith("humans")}.values(), 3000,
                   clave=lambda im: im["location"], por_grupo=20)
for im in vacias:
    agregar(fuente="wcs", id=im["id"], url=BASE_WCS + im["file_name"], split=split_por_ubicacion("wcs", im["location"]),
            ubic=str(im["location"]), cajas=[])
print(f"  {len(sel)} con animal, {len(vacias)} vacías ({time.time() - t0:.0f} s)")


# ---------------------------------------------------------------- 6. NACTI (cajas de MegaDetector)
print("NACTI (metadatos de 1.2 GB, tarda unos minutos)...")
BASE_NACTI = "https://storage.googleapis.com/public-datasets-lila/nacti-unzipped/"
d = json.load(open(META / "nacti_metadata.1.14.json", encoding="utf-8"))
cats = {c["id"]: c["name"].lower() for c in d["categories"]}
NAC = {"bos taurus": "bovino", "canis familiaris": "perro", "equus ferus": "caballo", "empty": "vacia"}
etq = defaultdict(set)
for a in d["annotations"]:
    etq[a["image_id"]].add(cats[a["category_id"]])
imgs = {i["id"]: i for i in d["images"]}
del d
grupos = defaultdict(list)
for i, s in etq.items():
    if len(s) == 1 and next(iter(s)) in NAC:        # solo fotos con UNA especie: sus cajas son de esa especie
        im = imgs[i]
        im["_especie"] = NAC[next(iter(s))]
        grupos[im["_especie"]].append(im)
del etq, imgs
quiero = {im["file_name"]: im for im in
          muestrear(grupos["bovino"], 9000, clave=lambda im: im["location"], por_grupo=80)
          + grupos["perro"] + grupos["caballo"]
          + muestrear(grupos["vacia"], 4000, clave=lambda im: im["location"], por_grupo=40)}
del grupos
print(f"  {len(quiero)} candidatas; leyendo las cajas de MegaDetector (1.5 GB)...")
md = json.load(open(META / "nacti_mdv5a.0.0_results.json", encoding="utf-8"))
n_nacti = Counter()
for r in md["images"]:
    im = quiero.get(r["file"])
    if im is None:
        continue
    dets = [x for x in (r.get("detections") or []) if x["category"] == "1"]
    seguros = [x for x in dets if x["conf"] >= 0.5]
    dudosos = [x for x in dets if 0.15 <= x["conf"] < 0.5]
    sp = split_por_ubicacion("nacti", im["location"])
    if im["_especie"] == "vacia":
        if seguros or dudosos:            # MegaDetector ve algo en una foto "vacía": se descarta
            continue
        cajas = []
    else:
        if not seguros:                   # sin ninguna caja segura: se descarta
            continue
        cajas = [(im["_especie"], b[0], b[1], b[0] + b[2], b[1] + b[3]) for b in (x["bbox"] for x in seguros)]
    # las cajas dudosas van como "otras": no se usan de objetivo ni para negativos difíciles
    otras = [("dudoso", b[0], b[1], b[0] + b[2], b[1] + b[3]) for b in (x["bbox"] for x in dudosos)]
    agregar(fuente="nacti", id=im["id"], url=BASE_NACTI + im["file_name"], split=sp, ubic=im["location"],
            cajas=cajas, otras=otras)
    n_nacti[im["_especie"]] += 1
del md, quiero
print(f"  NACTI: {dict(n_nacti)} ({time.time() - t0:.0f} s)")

# ---------------------------------------------------------------- 7. Tope de fotos completas en val/test
# Para que evaluar no tarde horas: máximo 350 fotos por (fuente, etiqueta principal) en val y en test.
cuenta, finales = Counter(), []
rng.shuffle(registros)
for r in registros:
    if r["split"] != "train":
        clave = (r["split"], r["fuente"], r["etiquetas"][0] if r["etiquetas"] else "fondo")
        if cuenta[clave] >= 350:
            continue
        cuenta[clave] += 1
    finales.append(r)
registros = finales

tabla = Counter((r["split"], r["fuente"], "+".join(r["etiquetas"]) or "fondo") for r in registros)
print("\nFotos por split / fuente / etiqueta:")
for (sp, f, e), n in sorted(tabla.items()):
    print(f"  {sp:5s} {f:10s} {e:22s} {n:6d}")
if args.solo_listar:
    sys.exit(0)


# ---------------------------------------------------------------- 8. Descargar y recortar
def descargar(url):
    for intento in range(6):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                          timeout=40).read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(2 ** intento)
        except Exception:
            time.sleep(2 ** intento)
    return None


def a_pixeles(c, W, H):
    return c[1] * W, c[2] * H, c[3] * W, c[4] * H


def interseccion(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h


def etiquetar(ventana, cajas_px, lado):
    """Clases visibles en la ventana. None si hay un animal 'a medias' (muy cortado o diminuto)."""
    etiquetas = set()
    for clase, b in cajas_px:
        area = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        frac = interseccion(ventana, b) / area
        tam = max(b[2] - b[0], b[3] - b[1]) * LADO_TRAIN / lado     # tamaño en el recorte final
        if frac >= 0.5 and tam >= 10:
            etiquetas.add(clase)
        elif frac >= 0.2:
            return None
    return etiquetas


def recorte_contexto(W, H, b, r):
    """Cuadrado k veces más grande que la caja b, con la caja adentro y en posición al azar."""
    bw, bh = b[2] - b[0], b[3] - b[1]
    k = math.exp(r.uniform(math.log(1.3), math.log(8.0)))
    lado = min(max(k * max(bw, bh), 64), W, H)
    def pos(lo_caja, hi_caja, total):
        lo, hi = max(0, hi_caja - lado), min(total - lado, lo_caja)
        return r.uniform(lo, hi) if lo <= hi else min(max((lo_caja + hi_caja - lado) / 2, 0), total - lado)
    x0, y0 = pos(b[0], b[2], W), pos(b[1], b[3], H)
    return (x0, y0, x0 + lado, y0 + lado), k


def recorte_azar(W, H, r, minimo=0.25, maximo=1.0):
    lado = r.uniform(minimo, maximo) * min(W, H)
    x0, y0 = r.uniform(0, W - lado), r.uniform(0, H - lado)
    return (x0, y0, x0 + lado, y0 + lado)


def guardar(img, ventana, ruta):
    rec = img.crop(tuple(round(v) for v in ventana)).resize((LADO_TRAIN, LADO_TRAIN), Image.BILINEAR)
    rec.save(ruta, quality=90)


def procesar(r):
    nombre = f"{r['fuente']}_{str(r['id']).replace('/', '_')}"
    carpeta = DESTINO / r["split"]
    hecho = carpeta / f"{nombre}.listo"
    if hecho.exists():
        return json.loads(hecho.read_text())
    datos = descargar(r["url"])
    if datos is None:
        return [("_error", "descarga", 0.0)]
    try:
        img = Image.open(io.BytesIO(datos)).convert("RGB")
    except Exception:
        return [("_error", "imagen_danada", 0.0)]
    W, H = img.size
    filas = []
    rr = random.Random(f"{SEED}|{nombre}")
    if r["split"] != "train":                    # foto completa
        img.thumbnail((LADO_EVAL, LADO_EVAL), Image.BILINEAR)
        img.save(carpeta / f"{nombre}.jpg", quality=92)
        filas.append((f"{nombre}.jpg", r["etiquetas"], 0.0))
    else:
        cajas = [(c[0], a_pixeles(c, W, H)) for c in r["cajas"]]
        otras = [a_pixeles(c, W, H) for c in r["otras"]]
        objetivos = [c for c in cajas if max(c[1][2] - c[1][0], c[1][3] - c[1][1]) >= 16]
        camara = r["fuente"] in ("ena24", "cct", "wcs", "nacti")
        if objetivos:
            n = 2 if (camara and ("caballo" in r["etiquetas"] or r["fuente"] == "ena24")) else 1
            for j in range(n):
                for _ in range(8):
                    ventana, k = recorte_contexto(W, H, rr.choice(objetivos)[1], rr)
                    et = etiquetar(ventana, cajas, ventana[2] - ventana[0])
                    if et:
                        guardar(img, ventana, carpeta / f"{nombre}_{j}.jpg")
                        filas.append((f"{nombre}_{j}.jpg", sorted(et), round(k, 2)))
                        break
            if camara and rr.random() < 0.3:          # negativo difícil: misma escena, sin animal
                todas = [c[1] for c in cajas] + otras
                for _ in range(10):
                    v = recorte_azar(W, H, rr, 0.25, 0.7)
                    if all(interseccion(v, b) < 0.05 * (b[2] - b[0]) * (b[3] - b[1]) for b in todas):
                        guardar(img, v, carpeta / f"{nombre}_n.jpg")
                        filas.append((f"{nombre}_n.jpg", [], 0.0))
                        break
        elif not r["cajas"] and not r["etiquetas"]:   # escena sin nuestros animales
            if otras and r["fuente"] == "openimages":
                v, k = recorte_contexto(W, H, otras[0], rr)
            else:
                v = recorte_azar(W, H, rr, 0.3, 1.0)
            guardar(img, v, carpeta / f"{nombre}_0.jpg")
            filas.append((f"{nombre}_0.jpg", [], 0.0))
    hecho.write_text(json.dumps(filas))
    return filas


for sp in ("train", "val", "test"):
    (DESTINO / sp).mkdir(parents=True, exist_ok=True)
filas_manifest, errores, t1 = [], Counter(), time.time()
with ThreadPoolExecutor(args.hilos) as pool:
    futuros = {pool.submit(procesar, r): r for r in registros}
    for n, fut in enumerate(as_completed(futuros), 1):
        r = futuros[fut]
        try:
            salida = fut.result()
        except Exception as e:
            salida = [("_error", type(e).__name__, 0.0)]
        for archivo, et, k in salida:
            if archivo == "_error":
                errores[(r["fuente"], et)] += 1
                continue
            fila = {"split": r["split"], "archivo": f"{r['split']}/{archivo}", "fuente": r["fuente"],
                    "id": r["id"], "ubicacion": r["ubic"], "k_contexto": k}
            fila.update({a: int(a in et) for a in ANIMALES})
            filas_manifest.append(fila)
        if n % 2000 == 0 or n == len(registros):
            print(f"  {n}/{len(registros)} fotos procesadas, {len(filas_manifest)} imágenes, "
                  f"{(time.time() - t1) / 60:.1f} min, errores {sum(errores.values())}", flush=True)

man = pd.DataFrame(filas_manifest).sort_values("archivo")
man.to_csv(DESTINO / "manifest.csv", index=False)
print("\nErrores:", dict(errores))
man["etiqueta"] = man[ANIMALES].apply(lambda f: "+".join(a for a in ANIMALES if f[a]) or "fondo", axis=1)
print("\nImágenes por split y etiqueta:")
print(man.groupby(["split", "etiqueta"]).size().unstack(0).fillna(0).astype(int))
print(f"\nListo en {(time.time() - t0) / 60:.1f} min -> {DESTINO}")
