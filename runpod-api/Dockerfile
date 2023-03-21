FROM runpod/stable-diffusion:web-automatic-3.0.0

SHELL ["/bin/bash", "-c"]

ENV PATH="${PATH}:/workspace/stable-diffusion-webui/venv/bin"

WORKDIR /

RUN rm /workspace/stable-diffusion-webui/models/Stable-diffusion/SDv1-5.ckpt
RUN rm /workspace/stable-diffusion-webui/models/Stable-diffusion/SDv2-768.ckpt
RUN mkdir /workspace/stable-diffusion-webui/models/Lora
RUN wget -O /workspace/stable-diffusion-webui/models/Lora/Lucario.safetensors \
    https://civitai.com/api/download/models/16734 --content-disposition
RUN wget -O /workspace/stable-diffusion-webui/models/Stable-diffusion/anything-v4.5-pruned.safetensors \
    https://huggingface.co/andite/anything-v4.0/resolve/main/anything-v4.5-pruned.safetensors
RUN cd /workspace/stable-diffusion-webui && git fetch && git pull

RUN pip install -U xformers
RUN pip install runpod

ADD handler.py .
ADD start.sh /start.sh
RUN chmod +x /start.sh

CMD [ "/start.sh" ]