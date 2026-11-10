# ══════════════════════════════════════════════════════════════════════════════
# markahix/receptor-library-manager
# CPU container for receptor preparation and library management.
# GPU not required (autogrid4 is CPU-only).
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim

LABEL description="Receptor library manager for AutoDock pipeline"
LABEL maintainer="markahix"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsqlite3-dev \
        wget \
        openbabel \
        python3-openbabel \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages ───────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
        "numpy<2,>=1.23" \
        pdbfixer \
        openmm \
        biopython

# ── AutoGrid4 precompiled binary ──────────────────────────────────────────────
RUN wget -q -O /tmp/autodocksuite.tar \
    "https://autodock.scripps.edu/wp-content/uploads/sites/56/2021/10/autodocksuite-4.2.6-x86_64Linux2.tar" \
    && tar -xf /tmp/autodocksuite.tar -C /tmp \
    && cp /tmp/x86_64Linux2/autogrid4 /usr/local/bin/autogrid4 \
    && chmod +x /usr/local/bin/autogrid4 \
    && rm -rf /tmp/autodocksuite.tar /tmp/x86_64Linux2

# ── AD4 parameters ────────────────────────────────────────────────────────────
RUN mkdir -p /usr/share/autodock
COPY AD4_parameters.dat /usr/share/autodock/AD4_parameters.dat
ENV AD4_PARAMETER_FILE=/usr/share/autodock/AD4_parameters.dat

# ── Application scripts ───────────────────────────────────────────────────────
WORKDIR /app
ARG CACHEBUST=1
COPY *.py ./

# ── Runtime directories ───────────────────────────────────────────────────────
RUN mkdir -p /receptors/inbox

ENTRYPOINT ["python3"]
CMD ["--help"]
