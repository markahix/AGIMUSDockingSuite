# ══════════════════════════════════════════════════════════════════════════════
# markahix/ligand-library-manager
# CPU-only container for ligand library management.
# No GPU required. Manages library state via SQLite.
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim

LABEL description="Ligand library manager for AutoDock pipeline"
LABEL maintainer="markahix"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages ───────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
        "numpy<2,>=1.23" \
        rdkit \
        biopython

# ── Application scripts ───────────────────────────────────────────────────────
WORKDIR /app
ARG CACHEBUST=1
COPY *.py ./

# ── Runtime directories ───────────────────────────────────────────────────────
RUN mkdir -p /ligands /library

# ── Default: show help ────────────────────────────────────────────────────────
ENTRYPOINT ["python3"]
CMD ["--help"]
