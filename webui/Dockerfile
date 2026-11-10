# ══════════════════════════════════════════════════════════════════════════════
# markahix/autodock-webui
# Lightweight orchestrator UI. No GPU, no scientific computation.
# Reads state databases from shared volumes, submits job files.
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim

LABEL description="AutoDock pipeline web UI orchestrator"
LABEL maintainer="markahix"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages ───────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
        flask \
        flask-socketio \
        simple-websocket

# ── Application ───────────────────────────────────────────────────────────────
WORKDIR /app
ARG CACHEBUST=1
COPY *.py ./
COPY templates/ ./templates/

# ── Runtime directories (mounted from host) ───────────────────────────────────
RUN mkdir -p /ligands /library /receptors /jobs/pending /jobs/running \
             /jobs/completed /jobs/failed

EXPOSE 5000

ENTRYPOINT ["python3"]
CMD ["server.py"]
