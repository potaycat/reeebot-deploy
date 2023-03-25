FROM python:3.10-slim-bullseye

RUN apt update
# for image module
RUN apt install -y ffmpeg libsm6 libxext6 git

RUN git clone https://github.com/potaycat/Reeebot-a-Discord-bot src
WORKDIR src
COPY .env .
COPY data ./data
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# RUN python tests.py

CMD ["python"]
