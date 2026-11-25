# historial/ – Experimentos previos (no forman parte del modelo final)

Aquí están los experimentos que **llevaron** al modelo final, pero que el modelo final no usa.
Se conservan porque son la evidencia de cada decisión (el profesor puede preguntar por qué se
pasó de una versión a otra). Todo está explicado en `docs/DOCUMENTACION.md`.

| Experimento | Script | Resultado en test | Qué nos enseñó |
|---|---|---|---|
| E1 | `src/exp1_explorar_datos.py` | – | 5 clases × 500 fotos, tamaños variables |
| E2 | `src/exp2_entrenar_cnn_v1.py` | 58.8 % | La CNN básica (como en MNIST) no alcanza |
| E3 | `src/exp3_entrenar_cnn_v2.py` | 68.8 % ± 2.8 (3 semillas); 79.4 % con ~2 300 fotos/clase | BatchNorm + GlobalAveragePooling = la arquitectura que usa el modelo final |
| E4 | `src/exp4_curva_aprendizaje.py` | 68.8 → 70.4 → 74.8 → 79.4 % | Más datos sigue ayudando → se armó el dataset máximo |

- `modelos/`: las redes de E2, E3 y E4 (`cnn_propia*.keras`).
- `resultados/`: sus logs, curvas, matrices de confusión, errores y
  `evaluacion_todos_los_modelos.txt` (la comparación de todos los modelos en el mismo test).
- `data/`: subconjuntos de 1 000 y 1 500 fotos por clase de la curva de aprendizaje (E4).
- `src/comun.py`: copia de `comun.py` tal como estaba al correr estos experimentos.

Para volver a correr uno (desde la carpeta del proyecto):

```powershell
.venv\Scripts\python.exe historial\src\exp2_entrenar_cnn_v1.py
```
