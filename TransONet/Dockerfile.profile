FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        graphviz \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-profile.txt /tmp/requirements-profile.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/requirements-profile.txt \
    && python -m pip install torch-scatter \
        -f https://data.pyg.org/whl/torch-2.7.1+cu128.html

WORKDIR /workspace

CMD ["bash"]
