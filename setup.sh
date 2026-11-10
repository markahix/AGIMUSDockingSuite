#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# setup.sh — AutoDock Pipeline One-Time Installation
# Run once to build all images, create directory structure, and install
# the systemd job runner service.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh [--cuda-arch 86]
#
# Options:
#   --cuda-arch N   CUDA compute capability (default: 86 for RTX 3080/3080 Ti)
#   --no-gpu        Skip docking engine build (no GPU available)
#   --skip-build    Skip Docker builds (images already present)
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"  # parent of DockingSuite/
DATA_DIR="$BASE_DIR/data"
CUDA_ARCH=86
BUILD_GPU=true
SKIP_BUILD=false

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda-arch) CUDA_ARCH="$2"; shift 2 ;;
        --no-gpu)    BUILD_GPU=false; shift ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓${NC} $*"; }
warn() { echo -e "${YELLOW}  ⚠${NC} $*"; }
err()  { echo -e "${RED}  ✗${NC} $*"; exit 1; }
hdr()  { echo -e "\n${GREEN}══${NC} $*"; }

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   AutoDock Pipeline — Setup          ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "  Suite dir:  $SCRIPT_DIR"
echo "  Base dir:   $BASE_DIR"
echo "  Data dir:   $DATA_DIR"
echo "  CUDA arch:  $CUDA_ARCH"
echo "  Build GPU:  $BUILD_GPU"
echo ""

# These paths are derived from where setup.sh sits. Unpacking the suite in the
# wrong parent directory produces a plausible but wrong data root, so confirm
# before committing it to the config file every other component will trust.
if [ -t 0 ] && [ "${AUTODOCK_ASSUME_YES:-}" != "1" ]; then
    if [ -d "$DATA_DIR" ]; then
        echo "  (existing data tree found at $DATA_DIR)"
    else
        echo "  (no data tree at $DATA_DIR yet — it will be created)"
    fi
    read -r -p "  Is the data dir correct? [y/N] " _reply
    case "$_reply" in
        [yY]|[yY][eE][sS]) ;;
        *) echo ""
           echo "  Aborted. Move DockingSuite/ so that its parent is the"
           echo "  directory that should contain data/, then re-run setup.sh."
           exit 1 ;;
    esac
    echo ""
fi

# ── Prerequisites ─────────────────────────────────────────────────────────────
hdr "Checking prerequisites"

command -v docker &>/dev/null || err "Docker not found. Install Docker CE first."
ok "Docker: $(docker --version | head -1)"

if $BUILD_GPU; then
    docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 \
        nvidia-smi &>/dev/null \
        || { warn "GPU not accessible via Docker. Use --no-gpu to skip docking engine."; BUILD_GPU=false; }
    $BUILD_GPU && ok "GPU: accessible"
fi

python3 --version &>/dev/null || err "python3 not found"
ok "Python: $(python3 --version)"

# ── Directory structure ───────────────────────────────────────────────────────
hdr "Creating directory structure"

dirs=(
    "$DATA_DIR/ligands"
    "$DATA_DIR/library"
    "$DATA_DIR/receptors/inbox"
    "$DATA_DIR/receptors/processed"
    "$DATA_DIR/receptors/failed"
    "$DATA_DIR/jobs/pending"
    "$DATA_DIR/jobs/running"
    "$DATA_DIR/jobs/completed"
    "$DATA_DIR/jobs/failed"
)

for d in "${dirs[@]}"; do
    mkdir -p "$d"
done
ok "Data directories created under $DATA_DIR"

# ── Configuration file ────────────────────────────────────────────────────────
hdr "Writing configuration"

CONF_DIR="/etc/autodock"
CONF_FILE="$CONF_DIR/pipeline.conf"

if sudo mkdir -p "$CONF_DIR" 2>/dev/null; then
    CONF_SUDO=sudo
else
    CONF_DIR="$HOME/.config/autodock"
    CONF_FILE="$CONF_DIR/pipeline.conf"
    mkdir -p "$CONF_DIR"
    CONF_SUDO=""
    warn "Could not write /etc/autodock — using $CONF_FILE instead"
fi

# Single source of truth. run.sh and job_runner.sh both read this rather than
# deriving the data root from their own location.
${CONF_SUDO} tee "$CONF_FILE" > /dev/null << CONFEOF
# AutoDock Pipeline configuration
# Written by setup.sh on $(date '+%Y-%m-%d %H:%M:%S')
# Authoritative for run.sh and job_runner.sh. Edit here, not in the scripts.
AUTODOCK_BASE=$BASE_DIR
AUTODOCK_DATA=$DATA_DIR
AUTODOCK_SUITE=$SCRIPT_DIR
CONFEOF
ok "config written: $CONF_FILE"

# ── Docker images ─────────────────────────────────────────────────────────────
hdr "Building Docker images"

if ! $SKIP_BUILD; then
    # 1. Ligand library manager (CPU only — fast build)
    echo "  Building markahix/ligand-library-manager:0.4 …"
    docker build -t markahix/ligand-library-manager:0.4 \
        "$SCRIPT_DIR/ligand_library_manager" \
        || err "ligand-library-manager build failed"
    ok "markahix/ligand-library-manager:0.4"

    # 2. Receptor library manager (CPU — autogrid4)
    echo "  Building markahix/receptor-library-manager:0.4 …"
    docker build -t markahix/receptor-library-manager:0.4 \
        "$SCRIPT_DIR/receptor_library_manager" \
        || err "receptor-library-manager build failed"
    ok "markahix/receptor-library-manager:0.4"

    # 3. Docking engine (GPU — longest build)
    if $BUILD_GPU; then
        echo "  Building markahix/docking-engine:0.4 (CUDA_ARCH=$CUDA_ARCH) …"
        echo "  This may take 10-20 minutes (compiling AutoDock-GPU from source)"
        docker build \
            --build-arg CUDA_ARCH="$CUDA_ARCH" \
            -t markahix/docking-engine:0.4 \
            "$SCRIPT_DIR/docking_engine" \
            || err "docking-engine build failed"
        ok "markahix/docking-engine:0.4 (sm_$CUDA_ARCH)"
    else
        warn "Skipping docking-engine (no GPU)"
    fi

    # 4. Web UI (lightweight — fast build)
    echo "  Building markahix/autodock-webui:0.4 …"
    docker build -t markahix/autodock-webui:0.4 \
        "$SCRIPT_DIR/webui" \
        || err "autodock-webui build failed"
    ok "markahix/autodock-webui:0.4"
else
    warn "Skipping builds (--skip-build)"
fi

# ── Systemd job runner ────────────────────────────────────────────────────────
hdr "Installing systemd job runner"

sudo mkdir -p /opt/autodock

# Installed verbatim — no sed-patching of paths. The runner reads the config
# file instead, so the installed copy stays byte-identical to the suite copy
# and 'run.sh doctor' can diff them to detect a stale install.
sudo cp "$SCRIPT_DIR/systemd/job_runner.sh" /opt/autodock/job_runner.sh
sudo chmod +x /opt/autodock/job_runner.sh
ok "job_runner.sh installed to /opt/autodock/"

# Only the config path is templated into the unit.
sed "s|AUTODOCK_CONF=/etc/autodock/pipeline.conf|AUTODOCK_CONF=$CONF_FILE|g" \
    "$SCRIPT_DIR/systemd/autodock-job-runner.service" \
    | sudo tee /etc/systemd/system/autodock-job-runner.service > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable --now autodock-job-runner
ok "autodock-job-runner service enabled and started"

sleep 2
if systemctl is-active --quiet autodock-job-runner; then
    ok "Job runner is running"
else
    warn "Job runner did not start. Check: journalctl -u autodock-job-runner -n 30"
fi

hdr "Verifying installation"
"$SCRIPT_DIR/run.sh" doctor || true

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Setup complete!                                    ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║                                                      ║"
echo "║   Run:   ./run.sh --help                             ║"
echo "║   UI:    ./run.sh ui  → http://localhost:5000        ║"
echo "║   Dock:  ./run.sh dock <receptor_name>               ║"
echo "║                                                      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
