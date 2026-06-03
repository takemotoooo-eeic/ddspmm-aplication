FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git fluidsynth libsndfile1 \
        build-essential cython3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip "setuptools>=61,<81" wheel && pip3 install uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

ENV UV_COMPILE_BYTECODE=1
ENV UV_PREVIEW_FEATURES=extra-build-dependencies

RUN uv export --frozen --no-dev --no-emit-workspace -o /tmp/requirements.txt && \
    uv pip install --system -r /tmp/requirements.txt

COPY . .

ENV PYTHONPATH=/app

RUN test -f api/models/fluidsynth_assets/soundfonts/FluidR3_GM.sf2 || \
    (echo "Missing: api/models/fluidsynth_assets/soundfonts/FluidR3_GM.sf2" && exit 1)

RUN uv pip install --system --no-deps -e . && \
    apt-get update && \
    apt-get purge -y --auto-remove build-essential cython3 python3.10-dev && \
    rm -rf /var/lib/apt/lists/*
