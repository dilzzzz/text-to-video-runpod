FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

WORKDIR /app

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
ENV DEBIAN_FRONTEND=noninteractive

# Python 3.12 + system deps
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    python3-venv \
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

# Virtual environment use කරලා pip install (system pip conflict avoid)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

# PyTorch 2.7 CUDA 12.8
RUN pip install \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# LTX-2.3 official packages
RUN pip install \
    "git+https://github.com/Lightricks/LTX-2.git#subdirectory=ltx-core" \
    "git+https://github.com/Lightricks/LTX-2.git#subdirectory=ltx-pipelines"

# Supporting deps
RUN pip install \
    huggingface_hub>=0.30.0 \
    transformers>=4.51.0 \
    accelerate>=1.2.0 \
    safetensors>=0.4.4

# RunPod + utils
RUN pip install \
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
