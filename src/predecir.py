"""
predecir.py - Identificar el animal en una o varias fotos (lo que se muestra en la demo).

Uso:
  .venv\\Scripts\\python.exe src\\predecir.py fotos_prueba\\vaca.jpg
  .venv\\Scripts\\python.exe src\\predecir.py fotos_prueba                 (toda una carpeta)
  .venv\\Scripts\\python.exe src\\predecir.py foto.jpg --mostrar           (abre una ventana)
  .venv\\Scripts\\python.exe src\\predecir.py foto.jpg --modelo modelos\\cnn_propia.keras

Para cada foto imprime la clase, la confianza y las 5 probabilidades, y guarda una copia
con la etiqueta dibujada en resultados/predicciones/.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from clasificador import MODELO_POR_DEFECTO, RAIZ, ClasificadorAnimales

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def listar_imagenes(rutas):
    for r in map(Path, rutas):
        if r.is_dir():
            yield from sorted(p for p in r.iterdir() if p.suffix.lower() in EXTENSIONES)
        elif r.exists():
            yield r
        else:
            print(f"[!] No existe: {r}")


def dibujar_etiqueta(img, texto, color):
    img = img.copy()
    d = ImageDraw.Draw(img)
    tam = max(16, img.width // 25)
    try:
        fuente = ImageFont.truetype("arial.ttf", tam)
    except OSError:
        fuente = ImageFont.load_default()
    caja = d.textbbox((0, 0), texto, font=fuente)
    d.rectangle([0, 0, caja[2] + 16, caja[3] + 12], fill=color)
    d.text((8, 4), texto, fill="white", font=fuente)
    return img


def main():
    ap = argparse.ArgumentParser(description="Clasifica animales en fotos con la CNN entrenada")
    ap.add_argument("rutas", nargs="+", help="archivos de imagen o carpetas")
    ap.add_argument("--modelo", default=str(MODELO_POR_DEFECTO))
    ap.add_argument("--umbral", type=float, default=0.60, help="confianza mínima (0-1)")
    ap.add_argument("--mostrar", action="store_true", help="mostrar cada imagen en una ventana")
    ap.add_argument("--rapido", action="store_true", help="un solo modelo, sin espejo (para tiempo real)")
    args = ap.parse_args()

    clf = ClasificadorAnimales(args.modelo, umbral=args.umbral, rapido=args.rapido)
    salida = RAIZ / "resultados" / "predicciones"
    salida.mkdir(parents=True, exist_ok=True)

    for ruta in listar_imagenes(args.rutas):
        img = Image.open(ruta).convert("RGB")
        r = clf.predecir(np.array(img))

        if r["incierto"]:
            texto, color = f"¿{r['nombre']}? {r['confianza']:.0%} (incierto)", (200, 140, 0)
        elif r["es_animal"]:
            texto, color = f"{r['nombre']} {r['confianza']:.0%}", (200, 30, 30)
        else:
            texto, color = f"{r['nombre']} {r['confianza']:.0%}", (60, 60, 60)

        print(f"\n{ruta.name}: {texto}")
        for c, p in sorted(r["probabilidades"].items(), key=lambda kv: -kv[1]):
            print(f"   {c:<8} {p:6.1%}  {'█' * int(p * 30)}")

        anotada = dibujar_etiqueta(img, texto, color)
        anotada.save(salida / f"pred_{ruta.stem}.jpg", quality=92)
        if args.mostrar:
            anotada.show()

    print(f"\nImágenes anotadas guardadas en: {salida}")


if __name__ == "__main__":
    main()
