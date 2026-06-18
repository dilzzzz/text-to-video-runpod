FROM python:3.11-slim

WORKDIR /app

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 git wget curl gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install diffusers transformers accelerate huggingface_hub safetensors

RUN pip install runpod Pillow numpy imageio imageio-ffmpeg

COPY . .

CMD ["python", "-u", "handler.py"]
