FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-distutils python3-pip git && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && pip3 install uv

WORKDIR /app

COPY . .

ENV PYTHONPATH=/app

RUN uv pip install --system -e .
