# Proyecto 6 – Detección de animales en vías y zonas rurales (CNN)

Clasificador de imágenes con una **red neuronal convolucional propia, entrenada desde cero**
(TensorFlow/Keras, sin modelos preentrenados) que identifica **bovino, caballo, gato, perro** o
**fondo** (sin animal) en una foto. Es la primera fase del sistema: el modelo queda listo para
conectarse a una cámara o video.

📘 **Documentación completa paso a paso:** [docs/DOCUMENTACION.md](docs/DOCUMENTACION.md)

## El modelo final

**Ensemble de 3 CNN propias + TTA → 95.2 % en test (476/500 fotos nunca vistas).**

- `modelos/modelo_final.json` indica qué redes usar: `cnn_propia_v2_receta.keras`, `_s1` y `_s2`.
- Las 3 tienen la misma arquitectura (4 bloques Conv-BN-ReLU ×2 + MaxPool, GlobalAveragePooling,
  Dropout, Dense de 5; 1.18 M de parámetros) y se entrenaron desde cero con 27 202 fotos a 224 px.
  Solo cambia la semilla.
- Cada foto se pasa por las 3 redes, normal y en espejo (TTA), y se promedian las probabilidades.

| Cómo se llegó (mismo test de 500 fotos) | Accuracy |
|---|---|
| E2: CNN v1 (500 fotos/clase) | 58.8 % |
| E3: CNN v2, con BatchNorm + GlobalAveragePooling (500 fotos/clase) | 68.8 % ± 2.8 (3 semillas) |
| E3: CNN v2 + dataset ampliado (~2 300 fotos/clase) | 79.4 % |
| Pasos 1–3: CNN v2 + receta moderna + dataset máximo, 224 px | 93.1 % ± 1.2 (3 semillas) |
| **Paso 4: modelo final, ensemble de 3 + TTA** | **95.2 %** |

E1–E4 son experimentos previos y están en [historial/](historial/LEEME.md).
`src/` tiene solo los 5 pasos del modelo final.

## Uso rápido

```powershell
# instalar (solo la primera vez)
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

# identificar el animal en una foto o en una carpeta de fotos
.venv\Scripts\python.exe src\predecir.py fotos_prueba
.venv\Scripts\python.exe src\predecir.py fotos_prueba --rapido     # 1 modelo, ~50 ms por imagen
```

## Datos (no están en el repo)

Los datos pesan ~4 GB, así que no se suben a GitHub. **Para la demo no hacen falta**: los modelos
ya entrenados están en `modelos/`. Para reentrenar, cada archivo va dentro de `data/`:

| Qué | Para qué | De dónde |
|---|---|---|
| `data/dataset_cnn_maximo/` (1.2 GB) | Entrenar y evaluar directo (pasos 3, 4 y 5) | Drive del grupo: `dataset_cnn_maximo.zip` |
| `data/dataset_cnn/` (93 MB) | Dataset original del curso; base del paso 1 | Drive del grupo: `datos_base.zip` |
| `data/anotaciones_open_images/open_images_animales_{train,val}.csv` (11 MB) | Cajas filtradas de nuestras clases (paso 1) | Drive del grupo: `datos_base.zip` |
| `data/anotaciones_open_images/oidv6-train-annotations-bbox.csv` (2.2 GB) | Pasos 1 y 2 | [Open Images (oficial)](https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv) |
| `data/anotaciones_open_images/test-annotations-bbox.csv` (77 MB) | Paso 2 | [Open Images (oficial)](https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv) |
| `data/anotaciones_open_images/validation-annotations-bbox.csv` (25 MB) | Referencia del test | [Open Images (oficial)](https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv) |
| `data/anotaciones_open_images/oidv7-class-descriptions-boxable.csv` | Nombres de las clases | [Open Images (oficial)](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv) |

Las imágenes son recortes de [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html)
(anotaciones CC BY 4.0; cada imagen conserva su licencia CC BY 2.0).

## Reentrenar el modelo final desde cero

```powershell
.venv\Scripts\python.exe src\paso1_ampliar_dataset.py       # ~2 300 fotos/clase de Open Images
.venv\Scripts\python.exe src\paso2_ampliar_al_maximo.py     # dataset máximo: 27 202 fotos
.venv\Scripts\python.exe src\paso3_entrenar_cnn.py          # GPU, ~75 min (semilla 42)
$env:SEMILLA = "1"; .venv\Scripts\python.exe src\paso3_entrenar_cnn.py
$env:SEMILLA = "2"; .venv\Scripts\python.exe src\paso3_entrenar_cnn.py
Remove-Item Env:SEMILLA
.venv\Scripts\python.exe src\paso4_ensemble_tta.py          # arma modelos/modelo_final.json
.venv\Scripts\python.exe src\paso5_evaluar.py               # métricas, matriz de confusión, errores
```

## Usarlo desde otro código (cámara, video, API…)

```python
from clasificador import ClasificadorAnimales
clf = ClasificadorAnimales()              # preciso: ensemble + TTA (~300 ms en CPU)
clf = ClasificadorAnimales(rapido=True)   # rápido: 1 red (~50 ms en CPU)
r = clf.predecir_bgr(frame)               # frame de OpenCV
print(r["clase"], r["confianza"], r["es_animal"])
```

Ver la sección 14 de la documentación para la arquitectura propuesta de video.
