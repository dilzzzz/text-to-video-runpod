FROM python:3.11-slim

WORKDIR /app

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

# System deps + C++ build tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    wget \
    curl \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Pip upgrade
RUN pip install --upgrade pip

# PyTorch cu124
RUN pip install \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Core ML deps
RUN pip install \
    diffusers==0.33.1 \
    transformers==4.44.2 \
    accelerate==0.33.0 \
    huggingface_hub==0.24.6 \
    safetensors==0.4.4

# RunPod + utils
RUN pip install \
    runpod==1.7.4 \
    Pillow==10.4.0 \
    numpy==1.26.4 \
    scipy==1.13.0 \
    soundfile==0.12.1 \
    imageio==2.35.0 \
    imageio-ffmpeg==0.5.1

# Copy source
COPY . .

CMD ["python", "-u", "handler.py"]
