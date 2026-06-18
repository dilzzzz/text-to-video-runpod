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
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Core ML deps (separate installs to avoid conflicts)
RUN pip install diffusers==0.38.0
RUN pip install transformers==4.51.0
RUN pip install accelerate==1.2.0
RUN pip install huggingface_hub==0.30.0
RUN pip install safetensors==0.4.4

# RunPod + utils + av (required for LTX video export)
RUN pip install \
    runpod==1.7.4 \
    Pillow==10.4.0 \
    numpy==1.26.4 \
    scipy==1.13.0 \
    soundfile==0.12.1 \
    imageio==2.35.0 \
    imageio-ffmpeg==0.5.1 \
    av

# Copy source
COPY . .

CMD ["python", "-u", "handler.py"]
