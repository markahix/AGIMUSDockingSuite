# ══════════════════════════════════════════════════════════════════════════════
# markahix/docking-engine
# GPU-accelerated docking container.
# Requires NVIDIA GPU with compute capability >= 7.5.
# ══════════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

LABEL description="AutoDock-GPU docking engine"
LABEL maintainer="markahix"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        python3 \
        python3-pip \
        python3-dev \
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages ───────────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
        "numpy<2,>=1.23"

# ── Build AutoDock-GPU from source ────────────────────────────────────────────
# CUDA_ARCH: compute capability (default 86 = RTX 3080/3080 Ti)
# Override: docker build --build-arg CUDA_ARCH=89
ARG CUDA_ARCH=86

ENV GPU_INCLUDE_PATH=/usr/local/cuda/include
ENV GPU_LIBRARY_PATH=/usr/local/cuda/lib64

RUN git clone https://github.com/ccsb-scripps/AutoDock-GPU.git /build/AutoDock-GPU \
    && cd /build/AutoDock-GPU \
    && find . -name "*.h" -exec grep -l "MAX_NUM_OF_ROTBONDS" {} \; \
       | xargs -I{} sed -i 's/define MAX_NUM_OF_ROTBONDS.*/define MAX_NUM_OF_ROTBONDS 128/' {} \
    && find . -name "*.h" -exec grep -l "MAX_NUM_OF_ATOMS" {} \; \
       | xargs -I{} sed -i 's/define MAX_NUM_OF_ATOMS.*/define MAX_NUM_OF_ATOMS 512/' {} \
    && make DEVICE=CUDA NUMWI=128 TARGETS=${CUDA_ARCH} \
            GPU_INCLUDE_PATH=/usr/local/cuda/include \
            GPU_LIBRARY_PATH=/usr/local/cuda/lib64 \
    && cp bin/autodock_gpu_128wi /usr/local/bin/ \
    && chmod +x /usr/local/bin/autodock_gpu_128wi \
    && cd / && rm -rf /build

# ── Application scripts ───────────────────────────────────────────────────────
WORKDIR /app
ARG CACHEBUST=1
COPY *.py ./

# ── Runtime directories ───────────────────────────────────────────────────────
RUN mkdir -p /library /receptors

# Verify binary
RUN autodock_gpu_128wi --help 2>&1 | head -3 || true

ENTRYPOINT ["python3"]
CMD ["dock_batch.py", "--help"]
