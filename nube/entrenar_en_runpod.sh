#!/usr/bin/env bash
# Entrena una semilla de la CNN v3 en un Pod de Runpod (GPU en la nube).
#
# El internet de los servidores es mucho más rápido que la subida de una casa, así que
# NO se sube el dataset: se suben solo los scripts (src/) y la lista de fotos de Open Images
# (data/dataset_cnn_maximo/manifest_train.csv), y el Pod descarga y arma el dataset v3 con
# el MISMO paso 6 del PC. Sale idéntico: mismas fotos, mismos recortes (semillas fijas).
#
# Pasos:
#   1. Instala las mismas versiones de librerías que en el PC (PyTorch ya viene en la imagen).
#   2. Descarga las anotaciones públicas (Open Images, COCO, LILA: ENA24, Caltech, WCS, NACTI).
#   3. Arma data/dataset_v3 con src/paso6_armar_dataset_v3.py.
#   4. Entrena con src/paso7_entrenar_cnn_v3.py (igual que en el PC).
#
# Uso (dentro del Pod; el código copiado en /workspace/cnn):   bash nube/entrenar_en_runpod.sh 1      # 1 = semilla
# Resultado: modelos/cnn_v3_s1.keras + .json, resultados/log_paso7_cnn_v3_s1.txt
set -euo pipefail
SEMILLA="${1:?falta la semilla, por ejemplo: bash nube/entrenar_en_runpod.sh 1}"
# /workspace es un disco de RED (sobrevive si el Pod se detiene, pero es lento con miles de
# archivos chicos): los datos se arman en el disco LOCAL (/root/cnn) y en /workspace quedan
# solo los modelos y los logs, que sí deben sobrevivir.
mkdir -p /workspace/cnn/modelos /workspace/cnn/resultados /root/cnn
cp -r /workspace/cnn/src /workspace/cnn/nube /workspace/cnn/data /root/cnn/
cd /root/cnn
ln -sfn /workspace/cnn/modelos modelos
ln -sfn /workspace/cnn/resultados resultados
mkdir -p data/anotaciones_open_images data/externos/metadatos

echo "== $(date +%T) GPU y PyTorch de la imagen"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(0))"
nproc; free -g | head -2

echo "== $(date +%T) Librerías"
python -m pip install -q --root-user-action=ignore --break-system-packages "keras==3.15.1" "tensorflow-cpu==2.19.1" \
    pandas pillow opencv-python-headless scikit-learn matplotlib

echo "== $(date +%T) Anotaciones públicas (en paralelo)"
bajar() { [ -s "$2" ] || curl -sSL --retry 5 -o "$2" "$1"; }
OI=data/anotaciones_open_images; M=data/externos/metadatos; G=https://storage.googleapis.com/public-datasets-lila
bajar https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv $OI/oidv6-train-annotations-bbox.csv &
bajar https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv $OI/test-annotations-bbox.csv &
bajar https://s3.us-east-1.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip $M/coco.zip &
bajar $G/ena24/ena24.json $M/ena24.json &
bajar $G/caltechcameratraps/labels/caltech_bboxes_20200316.json $M/caltech_bboxes_20200316.json &
bajar $G/caltechcameratraps/labels/caltech_camera_traps.json.zip $M/caltech_camera_traps.json.zip &
bajar $G/wcs/wcs_20220205_bboxes_with_classes.zip $M/wcs_bboxes.zip &
bajar $G/nacti/nacti_metadata.1.14.json.zip $M/nacti_metadata.zip &
bajar https://lila.science/public/lila-md-results/nacti_mdv5a.0.0_results.json.zip $M/nacti_md.zip &
wait
python - <<'EOF'
import zipfile, pathlib
M = pathlib.Path("data/externos/metadatos")
for z in ["coco.zip", "caltech_camera_traps.json.zip", "wcs_bboxes.zip", "nacti_metadata.zip", "nacti_md.zip"]:
    with zipfile.ZipFile(M / z) as f:
        nombres = [n for n in f.namelist() if not n.endswith("/") and (z != "coco.zip" or "/instances_" in n)]
        f.extractall(M, members=nombres)
    (M / z).unlink()
print(sorted(p.name for p in M.rglob("*") if p.is_file()))
EOF

echo "== $(date +%T) Armando el dataset v3 (paso 6)"
python -u src/paso6_armar_dataset_v3.py --hilos 64 > resultados/log_paso6_armar_dataset_v3_nube.txt 2>&1
tail -22 resultados/log_paso6_armar_dataset_v3_nube.txt

echo "== $(date +%T) Entrenando semilla ${SEMILLA}"
KERAS_BACKEND=torch SEMILLA="${SEMILLA}" python -u src/paso7_entrenar_cnn_v3.py \
    > "resultados/log_paso7_cnn_v3_s${SEMILLA}.txt" 2>&1
echo "== $(date +%T) LISTO"
