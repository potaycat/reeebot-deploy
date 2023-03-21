#!/bin/bash
echo "Container Started"
export PYTHONUNBUFFERED=1
source /workspace/stable-diffusion-webui/venv/bin/activate
cd /workspace/stable-diffusion-webui

wget -ON workspace/stable-diffusion-webui/webui.py \
    https://gist.githubusercontent.com/potaycat/b3713e593dbe9b3570beea6239356f1d/raw/917b2c3fb6a8068ae39b6b688110391ef3eb767d/webui.py
echo "starting api"
python webui.py --port 3000 --nowebui --api --xformers \
    --ckpt models/Stable-diffusion/anything-v4.5-pruned.safetensors &
cd /

echo "starting worker"
python -u handler.py
