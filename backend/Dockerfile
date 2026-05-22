FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-distutils python3-pip git \
        fluidsynth && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip "setuptools>=61,<81" wheel && pip3 install uv

WORKDIR /app

COPY . .

ENV PYTHONPATH=/app

RUN test -f api/models/fluidsynth_assets/soundfonts/FluidR3_GM.sf2 || \
    (echo "Missing: api/models/fluidsynth_assets/soundfonts/FluidR3_GM.sf2" && exit 1)

RUN uv pip install --system "setuptools>=61,<81" wheel && \
    uv pip install --system -e .
