"""
DEMO EN VIDEO - La CNN propia sobre un video o la webcam, con OpenCV.

Modelos (--modelo):
  v3  (por defecto si existe modelos/cnn_v3.keras) CNN totalmente convolucional: recibe el
      frame COMPLETO a 640 px sin aplastarlo y su mapa de evidencias dice DÓNDE está cada animal.
  v2  el modelo anterior en modo rápido (1 red a 224x224).

Sirve para ver CÓMO aplicar el clasificador a video. Hay 3 formas, y se cambian en vivo:

  [1] COMPLETO    v2: el frame entero se reduce a 224x224 y se clasifica una vez (lo que
                  hacía la web; falla con animales lejanos: una vaca a 10 m queda en pocos
                  píxeles). v3: el frame va a 640 px de lado mayor y las cajas salen del
                  mapa de evidencias de la red.
  [2] MOSAICO     El frame se corta en ventanas cuadradas de 3 tamaños (con solapamiento)
                  y se clasifican todas en un solo lote. Una ventana pequeña "acerca" al
                  animal lejano. Es como una detección por fuerza bruta (ventana deslizante).
  [3] MOVIMIENTO  Para cámaras fijas: MOG2 (sustracción de fondo) encuentra lo que se mueve,
                  se recorta esa zona (cuadrada, con margen) y solo se clasifica el recorte.
                  Es la arquitectura de la sección 14 de la documentación.

Teclas:  1 / 2 / 3 cambiar de modo · espacio pausa · + / - umbral · q o Esc salir

Uso:
    .venv\\Scripts\\python.exe src\\demo_video.py videos_prueba\\vacas_carretera.mp4
    .venv\\Scripts\\python.exe src\\demo_video.py 0                       (webcam)
    .venv\\Scripts\\python.exe src\\demo_video.py video.mp4 --modelo v2 --modo 3 --guardar salida.mp4
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")     # PyTorch usa la GPU en Windows

import argparse
import sys
import time
from pathlib import Path

import cv2
import keras
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
CLASES = ["bovino", "caballo", "fondo", "gato", "perro"]
NOMBRES = {"bovino": "Bovino", "caballo": "Caballo", "fondo": "Fondo", "gato": "Gato", "perro": "Perro"}
COLORES = {"bovino": (0, 140, 255), "caballo": (200, 80, 200), "fondo": (160, 160, 160),
           "gato": (60, 200, 60), "perro": (255, 160, 0)}          # BGR
MODOS = {1: "COMPLETO", 2: "MOSAICO", 3: "MOVIMIENTO"}

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class MotorV2:
    """Modelo anterior, modo rápido = la red de semilla 42, sin espejo. Softmax de 5 clases."""
    nombre, img = "v2 (224 px)", 224

    def __init__(self):
        self.m = keras.models.load_model(RAIZ / "modelos" / "cnn_propia_v2_receta.keras")
        self.m(np.zeros((1, 224, 224, 3), np.float32), training=False)      # calentar la GPU

    def clasificar(self, recortes_rgb):
        """Recortes RGB uint8 -> probabilidades (N, 5) en el orden de CLASES. Bilineal sin
        mantener proporción, igual que en el entrenamiento; el modelo normaliza por dentro."""
        if not recortes_rgb:
            return np.zeros((0, len(CLASES)), np.float32)
        x = np.stack([cv2.resize(r, (self.img, self.img), interpolation=cv2.INTER_LINEAR) for r in recortes_rgb])
        return keras.ops.convert_to_numpy(self.m(x.astype(np.float32), training=False))

    def frame_completo(self, rgb, umbral):
        return self.clasificar([rgb])[0], None


class MotorV3:
    """CNN v3: 4 sigmoides (bovino, caballo, gato, perro) + mapa de evidencias por zona."""
    nombre, img = "v3 (frame 640 px)", 320
    ANIM = ["bovino", "caballo", "gato", "perro"]

    def __init__(self, archivo="cnn_v3.keras", lado=640):
        import capas  # noqa: F401  (registra la capa PoolingLSE)
        self.m = keras.models.load_model(RAIZ / "modelos" / archivo)
        self.mapa = keras.Model(self.m.input, [self.m.get_layer("mapa_evidencias").output, self.m.output])
        self.lado = lado
        self.m(np.zeros((1, 320, 320, 3), np.float32), training=False)

    @staticmethod
    def a_cinco(p4):
        """4 sigmoides -> vector de 5 en el orden de CLASES (fondo = 1 - el animal más probable)."""
        p4 = np.atleast_2d(p4)
        fondo = 1 - p4.max(axis=1, keepdims=True)
        return np.concatenate([p4[:, :2], fondo, p4[:, 2:]], axis=1)

    def clasificar(self, recortes_rgb):
        if not recortes_rgb:
            return np.zeros((0, len(CLASES)), np.float32)
        x = np.stack([cv2.resize(r, (self.img, self.img), interpolation=cv2.INTER_LINEAR) for r in recortes_rgb])
        return self.a_cinco(keras.ops.convert_to_numpy(self.m(x.astype(np.float32), training=False)))

    def frame_completo(self, rgb, umbral):
        """Frame a 640 px de lado mayor (sin aplastar) -> probabilidades + cajas del mapa."""
        H, W = rgb.shape[:2]
        f = self.lado / max(H, W)
        x = cv2.resize(rgb, (round(W * f), round(H * f)), interpolation=cv2.INTER_AREA).astype(np.float32)[None]
        mapa, p4 = self.mapa(x, training=False)
        mapa = 1 / (1 + np.exp(-keras.ops.convert_to_numpy(mapa)[0]))     # (h, w, 4) prob. por zona
        p4 = keras.ops.convert_to_numpy(p4)[0]
        cajas = []
        for j, a in enumerate(self.ANIM):
            if p4[j] < umbral:
                continue
            m = mapa[..., j]
            masc = cv2.resize((m >= umbral).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            n, _, stats, _ = cv2.connectedComponentsWithStats(masc)
            for k in range(1, n):
                x0, y0, w, h, _ = stats[k]
                zona = m[int(y0 * m.shape[0] / H):int((y0 + h) * m.shape[0] / H) + 1,
                         int(x0 * m.shape[1] / W):int((x0 + w) * m.shape[1] / W) + 1]
                cajas.append(((int(x0), int(y0), int(w), int(h)), a, float(zona.max())))
        return self.a_cinco(p4)[0], cajas


def cargar_modelo(cual):
    if cual == "v3" or (cual == "auto" and (RAIZ / "modelos" / "cnn_v3.keras").exists()):
        return MotorV3()
    return MotorV2()


def cuadrado(x, y, w, h, margen, W, H):
    """Caja -> caja cuadrada con margen, dentro del frame (para no deformar al animal)."""
    lado = int(max(w, h) * (1 + margen))
    lado = min(lado, W, H)
    cx, cy = x + w / 2, y + h / 2
    x0 = int(np.clip(cx - lado / 2, 0, W - lado))
    y0 = int(np.clip(cy - lado / 2, 0, H - lado))
    return x0, y0, lado, lado


def ventanas_mosaico(W, H):
    """Ventanas cuadradas de 3 tamaños (alto, 1/2 y 1/3 del alto) con 50 % de solapamiento."""
    cajas = []
    for div in (1, 2, 3):
        lado = H // div
        paso = lado // 2
        xs = list(range(0, W - lado + 1, paso)) or [0]
        ys = list(range(0, H - lado + 1, paso)) or [0]
        if xs[-1] != W - lado:
            xs.append(W - lado)
        if ys[-1] != H - lado:
            ys.append(H - lado)
        cajas += [(x, y, lado, lado) for y in ys for x in xs]
    return cajas


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def suprimir(detecciones, umbral_iou=0.3):
    """Non-maximum suppression simple: se queda con la caja más segura de cada zona."""
    detecciones = sorted(detecciones, key=lambda d: -d[2])
    final = []
    for d in detecciones:
        if all(iou(d[0], f[0]) < umbral_iou for f in final):
            final.append(d)
    return final


def dibujar_caja(img, caja, clase, conf):
    x, y, w, h = caja
    color = COLORES[clase]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
    texto = f"{NOMBRES[clase]} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(img, (x, max(0, y - th - 10)), (x + tw + 8, y), color, -1)
    cv2.putText(img, texto, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def dibujar_barras(img, probs, x0=10, y0=70):
    for i, (c, p) in enumerate(zip(CLASES, probs)):
        y = y0 + i * 28
        cv2.rectangle(img, (x0, y), (x0 + 300, y + 22), (40, 40, 40), -1)
        cv2.rectangle(img, (x0, y), (x0 + int(300 * p), y + 22), COLORES[c], -1)
        cv2.putText(img, f"{NOMBRES[c]:8s} {p:5.1%}", (x0 + 6, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="ruta del video, o 0 para la webcam")
    ap.add_argument("--modo", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--modelo", default="auto", choices=["auto", "v2", "v3"])
    ap.add_argument("--umbral", type=float, default=0.60)
    ap.add_argument("--guardar", help="guardar el video anotado en este .mp4")
    ap.add_argument("--sin-ventana", action="store_true", help="no abrir ventana (solo guardar / medir)")
    args = ap.parse_args()

    fuente = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir {args.video}")
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30

    print("Cargando la CNN...")
    modelo = cargar_modelo(args.modelo)
    print("Modelo:", modelo.nombre)
    mog2 = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    modo, umbral, pausa = args.modo, args.umbral, False
    suavizado = None                 # promedio móvil de probabilidades (modo COMPLETO)
    escritor = None
    nombre_ventana = "CNN propia - demo en video (1/2/3 modo, espacio pausa, q salir)"
    if not args.sin_ventana:
        cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    t_prev = time.perf_counter()
    conteo = {c: 0 for c in CLASES}
    n_frames = 0

    while True:
        if not pausa:
            ok, frame = cap.read()
            if not ok:
                if isinstance(fuente, int) or args.guardar or args.sin_ventana:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)     # el video se repite en bucle
                suavizado = None
                continue
            H, W = frame.shape[:2]
            if W > 1280:                                # limitar resolución de trabajo
                frame = cv2.resize(frame, (1280, int(H * 1280 / W)))
                H, W = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vista = frame.copy()
            mascara_mov = mog2.apply(frame)             # se actualiza siempre, así está listo al cambiar a 3

            t0 = time.perf_counter()
            if modo == 1:
                p, cajas_mapa = modelo.frame_completo(rgb, umbral)
                suavizado = p if suavizado is None else 0.6 * suavizado + 0.4 * p
                dibujar_barras(vista, suavizado)
                if cajas_mapa is None:          # v2: una sola respuesta para todo el frame
                    c = CLASES[int(np.argmax(suavizado))]
                    conf = float(suavizado.max())
                    if c != "fondo" and conf >= umbral:
                        dibujar_caja(vista, (4, 4, W - 8, H - 8), c, conf)
                        conteo[c] += 1
                else:                           # v3: cajas sacadas del mapa de evidencias
                    for caja, c, conf in cajas_mapa:
                        dibujar_caja(vista, caja, c, conf)
                    for c in {c for _, c, _ in cajas_mapa}:
                        conteo[c] += 1
                n_cajas = 1
            else:
                if modo == 2:
                    cajas = ventanas_mosaico(W, H)
                else:
                    m = np.where(mascara_mov == 255, 255, 0).astype(np.uint8)     # 127 = sombra
                    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, nucleo)
                    m = cv2.dilate(m, nucleo, iterations=3)
                    cont, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cajas = [cuadrado(*cv2.boundingRect(k), 0.3, W, H) for k in cont
                             if cv2.contourArea(k) > 0.0015 * W * H][:20]
                    peq = cv2.resize(m, (W // 5, H // 5))
                    vista[H - H // 5:, W - W // 5:] = cv2.cvtColor(peq, cv2.COLOR_GRAY2BGR)
                probs = modelo.clasificar([rgb[y:y + h, x:x + w] for x, y, w, h in cajas])
                dets = []
                for caja, p in zip(cajas, probs):
                    i = int(np.argmax(p))
                    if CLASES[i] != "fondo" and p[i] >= umbral:
                        dets.append((caja, CLASES[i], float(p[i])))
                    elif modo == 3:                       # en movimiento también se ve lo descartado
                        x, y, w, h = caja
                        cv2.rectangle(vista, (x, y), (x + w, y + h), (120, 120, 120), 1)
                for caja, c, conf in suprimir(dets):
                    dibujar_caja(vista, caja, c, conf)
                    conteo[c] += 1
                n_cajas = len(cajas)
            ms = (time.perf_counter() - t0) * 1000
            n_frames += 1

            ahora = time.perf_counter(); fps = 1 / max(ahora - t_prev, 1e-6); t_prev = ahora
            cv2.rectangle(vista, (0, 0), (W, 56), (20, 20, 20), -1)
            cv2.putText(vista, f"{modelo.nombre}  [{modo}] {MODOS[modo]}   CNN {ms:5.1f} ms ({n_cajas} recortes)   "
                               f"{fps:4.1f} fps   umbral {umbral:.2f}",
                        (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if args.guardar:
                if escritor is None:
                    escritor = cv2.VideoWriter(args.guardar, cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (W, H))
                escritor.write(vista)

        if args.sin_ventana:
            continue
        cv2.imshow(nombre_ventana, vista)
        tecla = cv2.waitKey(1 if not pausa else 30) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla in (ord("1"), ord("2"), ord("3")):
            modo = int(chr(tecla)); suavizado = None
        elif tecla == ord(" "):
            pausa = not pausa
        elif tecla in (ord("+"), ord("=")):
            umbral = min(0.95, umbral + 0.05)
        elif tecla == ord("-"):
            umbral = max(0.30, umbral - 0.05)
        if cv2.getWindowProperty(nombre_ventana, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    if escritor:
        escritor.release()
    cv2.destroyAllWindows()
    print(f"Frames: {n_frames}. Detecciones por clase (suma de frames): {conteo}")


if __name__ == "__main__":
    main()
