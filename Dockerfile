FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

WORKDIR /app

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

# Python 3.12 + system deps
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    wget \
    curl \
    gcc \
    g++ \
    cmake \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip --break-system-packages

# PyTorch 2.7 + CUDA 12.8
RUN pip install --break-system-packages \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# LTX-2.3 official packages from GitHub
RUN pip install --break-system-packages \
    "git+https://github.com/Lightricks/LTX-2.git#subdirectory=ltx-core" \
    "git+https://github.com/Lightricks/LTX-2.git#subdirectory=ltx-pipelines"

# Supporting deps
RUN pip install --break-system-packages \
    huggingface_hub>=0.30.0 \
    transformers>=4.51.0 \
    accelerate>=1.2.0 \
    safetensors>=0.4.4 \
    diffusers>=0.38.0

# RunPod + utils
RUN pip install --break-system-packages \
    runpod==1.7.4 \
    Pillow \
    numpy \
    scipy \
    soundfile \
    imageio \
    imageio-ffmpeg \
    av

COPY . .

CMD ["python", "-u", "handler.py"]
