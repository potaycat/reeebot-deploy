from python:3.10-buster

WORKDIR /

RUN pip install runpod

RUN pip install gdown
RUN gdown -O LucarioLoRA.safetensors https://drive.google.com/uc?id=1BrwBns8ktDh2J24XZmPppJwitigbZJ_N

RUN pip install transformers accelerate xformer diffusers==0.14.0 safetensors
RUN apt update && apt install -y git
RUN git clone https://github.com/kohya-ss/sd-scripts.git sdscripts

ADD handler.py .
ADD cache_models.py .

RUN python cache_models.py

CMD [ "python", "-u", "/handler.py" ]