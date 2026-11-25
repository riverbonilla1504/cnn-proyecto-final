"""
clasificador.py - Módulo REUTILIZABLE para usar el modelo entrenado.

Está separado del entrenamiento a propósito: quien implemente la cámara / video
solo necesita importar esta clase, sin tocar nada del entrenamiento.

Ejemplo con una foto:
    from clasificador import ClasificadorAnimales
    clf = ClasificadorAnimales()                 # carga modelos/modelo_final.keras
    r = clf.predecir_archivo("foto.jpg")
    print(r["clase"], r["confianza"])

El modelo final es un ENSEMBLE: 3 CNN propias (misma arquitectura, distinta semilla) y cada
foto se evalúa también volteada (TTA). Se promedian las 6 probabilidades. Todo eso lo
describe modelos/modelo_final.json; esta clase lo hace sola, quien la usa no nota diferencia.

Ejemplo con un frame de OpenCV (cámara o video):
    r = clf.predecir_bgr(frame)                  # OpenCV entrega BGR, aquí se convierte a RGB
    r = clf.predecir_bgr(frame[y1:y2, x1:x2])    # mejor: solo el recorte donde está el animal
"""
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
# Copia del mejor modelo PROPIO (CNN desde cero), elegido por validación; ver la documentación
MODELO_POR_DEFECTO = RAIZ / "modelos" / "modelo_final.keras"

NOMBRES_BONITOS = {"bovino": "Bovino (vaca/toro)", "caballo": "Caballo", "fondo": "Sin animal (fondo)",
                   "gato": "Gato", "perro": "Perro"}


class ClasificadorAnimales:
    def __init__(self, ruta_modelo=MODELO_POR_DEFECTO, umbral=0.60, rapido=False):
        """
        ruta_modelo: archivo .keras entrenado (al lado debe estar el .json con las clases).
        umbral: si la probabilidad máxima es menor que esto, el resultado se marca como
                "incierto". Útil en cámara para no lanzar alertas con predicciones dudosas.
        rapido: si el modelo es un ensemble + TTA (6 pasadas por imagen, ~310 ms en CPU),
                rapido=True usa solo el primer modelo y sin espejo (1 pasada, ~6 veces más
                rápido, ~1-2 puntos menos de accuracy). Pensado para video en tiempo real.
        """
        ruta_modelo = Path(ruta_modelo)
        meta = json.loads(ruta_modelo.with_suffix(".json").read_text(encoding="utf-8"))
        # "ensemble": lista de archivos .keras cuyas predicciones se promedian
        archivos = [ruta_modelo.parent / f for f in meta["ensemble"]] if "ensemble" in meta else [ruta_modelo]
        if rapido:
            archivos = archivos[:1]
        self.modelos = [tf.keras.models.load_model(f) for f in archivos]
        # "tta_volteo": cada imagen se evalúa también volteada horizontalmente (espejo)
        self.tta = bool(meta.get("tta_volteo", False)) and not rapido
        self.clases = meta["clases"]
        self.img_size = meta["img_size"]
        self.umbral = umbral

    # ------------------------------------------------------------------ preparación
    def _preparar(self, imagenes_rgb):
        """
        Lista de arrays RGB (alto, ancho, 3) uint8 -> tensor (N, img_size, img_size, 3) float32.
        Se redimensiona IGUAL que en el entrenamiento (bilineal, sin mantener proporción).
        La normalización NO se hace aquí: el modelo la trae incluida (capa Rescaling).
        """
        lote = [tf.image.resize(np.ascontiguousarray(np.asarray(im)[..., :3]),
                                (self.img_size, self.img_size),
                                method="bilinear") for im in imagenes_rgb]
        return tf.stack(lote)

    def _interpretar(self, probs):
        i = int(np.argmax(probs))
        clase = self.clases[i]
        conf = float(probs[i])
        return {
            "clase": clase,
            "nombre": NOMBRES_BONITOS.get(clase, clase),
            "confianza": conf,
            "es_animal": clase != "fondo" and conf >= self.umbral,
            "incierto": conf < self.umbral,
            "probabilidades": {c: float(p) for c, p in zip(self.clases, probs)},
        }

    # ------------------------------------------------------------------ API pública
    def predecir_lote(self, imagenes_rgb):
        """Varias imágenes RGB a la vez (más eficiente: una sola pasada por la red)."""
        if len(imagenes_rgb) == 0:
            return []
        x = self._preparar(imagenes_rgb)
        vistas = [x, tf.reverse(x, axis=[2])] if self.tta else [x]      # original + espejo
        # m(v) en vez de m.predict(v): para pocas imágenes (cámara) es bastante más rápido
        probs = np.mean([np.asarray(m(v, training=False)) for m in self.modelos for v in vistas], axis=0)
        return [self._interpretar(p) for p in probs]

    def predecir(self, imagen_rgb):
        """Una imagen RGB como numpy array (alto, ancho, 3)."""
        return self.predecir_lote([imagen_rgb])[0]

    def predecir_bgr(self, frame_bgr):
        """Para frames de OpenCV (cv2.imread / cv2.VideoCapture entregan BGR)."""
        return self.predecir(np.ascontiguousarray(frame_bgr[..., ::-1]))

    def predecir_archivo(self, ruta):
        with Image.open(ruta) as im:
            return self.predecir(np.array(im.convert("RGB")))
