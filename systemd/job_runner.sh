#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# job_runner.sh — AutoDock Pipeline Host-Side Job Runner
# Watches jobs/pending/ and executes one job at a time.
# Atomic mv prevents double-execution.
# Containers mirror host paths (no path translation needed).
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Configuration resolution ──────────────────────────────────────────────────
# The config file written by setup.sh is authoritative. Deriving the data root
# from the script's own location is a silent failure mode: an unpack in the
# wrong parent directory yields a valid-looking but empty second data tree.
# Config is consulted first; script-location derivation is only a fallback,
# and validate_paths() below refuses to proceed if the result is not a real
# pipeline data tree.
_autodock_find_conf() {
    local c
    for c in "${AUTODOCK_CONF:-}" \
             "/etc/autodock/pipeline.conf" \
             "${HOME:-/root}/.config/autodock/pipeline.conf"; do
        if [ -n "$c" ] && [ -r "$c" ]; then echo "$c"; return 0; fi
    done
    return 1
}

AUTODOCK_CONF_FILE="$(_autodock_find_conf || true)"
if [ -n "$AUTODOCK_CONF_FILE" ]; then
    # shellcheck disable=SC1090
    . "$AUTODOCK_CONF_FILE"
fi

BASE_DIR="${AUTODOCK_BASE:-}"
DATA_DIR="${AUTODOCK_DATA:-${BASE_DIR:+$BASE_DIR/data}}"
JOBS_DIR="${DATA_DIR:-}/jobs"
LOG_DIR="${AUTODOCK_LOG_DIR:-/var/log/autodock}"

if [ -z "${DATA_DIR:-}" ]; then
    echo "FATAL: no AutoDock config found and AUTODOCK_BASE is unset." >&2
    echo "       Expected /etc/autodock/pipeline.conf (written by setup.sh)." >&2
    exit 78   # EX_CONFIG
fi

# The runner must never create the queue it watches. If the tree is absent the
# data root is wrong, and silently creating an empty queue hides that.
for _d in "$DATA_DIR/library" "$DATA_DIR/receptors" "$JOBS_DIR"; do
    if [ ! -d "$_d" ]; then
        echo "FATAL: expected directory missing: $_d" >&2
        echo "       DATA_DIR=$DATA_DIR" >&2
        echo "       config=${AUTODOCK_CONF_FILE:-<none>}" >&2
        echo "       Run setup.sh, or correct the config file." >&2
        exit 78
    fi
done

LIGAND_IMAGE="markahix/ligand-library-manager:0.4"
RECEPTOR_IMAGE="markahix/receptor-library-manager:0.4"
DOCKING_IMAGE="markahix/docking-engine:0.4"

USER_ARG="--user $(id -u):$(id -g)"

mkdir -p "$LOG_DIR" "$JOBS_DIR/pending" "$JOBS_DIR/running" \
         "$JOBS_DIR/completed" "$JOBS_DIR/failed"

log_startup_paths() {
    log "config:   ${AUTODOCK_CONF_FILE:-<none — using AUTODOCK_BASE>}"
    log "data dir: $DATA_DIR"
    log "watching: $JOBS_DIR/pending"
}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/runner.log"; }

# ── Parse job field helper ─────────────────────────────────────────────────────
jget() { python3 -c "import json; d=json.load(open('$1')); print(d$2)" 2>/dev/null || echo ""; }
jopt() { python3 -c "import json; d=json.load(open('$1')); print(d.get('options',{}).get('$2',''))" 2>/dev/null || echo ""; }

# ══════════════════════════════════════════════════════════════════════════════
# Job handlers
# ══════════════════════════════════════════════════════════════════════════════

run_prep_ligands_job() {
    local job_file="$1"
    local job_id library_dir ligand_dir

    job_id=$(jget "$job_file" "['job_id']")
    library_dir=$(jget "$job_file" "['library_dir']")
    ligand_dir=$(jget  "$job_file" "['ligand_dir']")

    [[ -d "$ligand_dir"  ]] || { log "ERROR: ligand_dir not found: $ligand_dir";  return 1; }
    [[ -d "$library_dir" ]] || mkdir -p "$library_dir"

    local refresh="" conformers=""
    [[ "$(jopt "$job_file" refresh_all)" == "True" ]] && refresh="--refresh_all"
    local c; c=$(jopt "$job_file" conformers)
    [[ -n "$c" && "$c" != "None" && "$c" != "null" ]] && conformers="--conformers $c"

    log "Starting prep_ligands: $job_id"

    docker run --rm $USER_ARG \
        --network none \
        -v "$ligand_dir":"$ligand_dir":ro \
        -v "$library_dir":"$library_dir" \
        "$LIGAND_IMAGE" prep_ligand.py \
        --ligand_dir  "$ligand_dir" \
        --library_dir "$library_dir" \
        $refresh $conformers \
        >> "$LOG_DIR/${job_id}.log" 2>&1

    log "prep_ligands complete: $job_id"
}

run_prep_receptor_job() {
    local job_file="$1"
    local job_id pdb_path receptors_dir library_dir

    job_id=$(jget      "$job_file" "['job_id']")
    pdb_path=$(jget    "$job_file" "['pdb_path']")
    receptors_dir=$(jget "$job_file" "['receptors_dir']")
    library_dir=$(jget "$job_file" "['library_dir']")

    [[ -f "$pdb_path" ]]      || { log "ERROR: pdb_path not found: $pdb_path";           return 1; }
    [[ -d "$receptors_dir" ]] || { log "ERROR: receptors_dir not found: $receptors_dir"; return 1; }

    local cx cy cz padding rmsd
    cx=$(jopt      "$job_file" center_x)
    cy=$(jopt      "$job_file" center_y)
    cz=$(jopt      "$job_file" center_z)
    padding=$(jopt "$job_file" padding);      [[ -z "$padding" ]] && padding=10.0
    rmsd=$(jopt    "$job_file" rmsd_threshold); [[ -z "$rmsd" ]] && rmsd=0.5

    # Mount the directory containing the PDB so the container can read it
    local pdb_dir
    pdb_dir=$(dirname "$pdb_path")

    log "Starting prep_receptor: $job_id"
    log "  PDB: $pdb_path  centre: ($cx, $cy, $cz)"

    docker run --rm $USER_ARG \
        --network none \
        -v "$pdb_dir":"$pdb_dir":ro \
        -v "$receptors_dir":"$receptors_dir" \
        -v "$library_dir":"$library_dir":ro \
        -v "$(dirname "$job_file")":"$(dirname "$job_file")":ro \
        "$RECEPTOR_IMAGE" prep_receptor.py \
        --job "$job_file" \
        >> "$LOG_DIR/${job_id}.log" 2>&1

    # Move PDB to processed
    local processed_dir="$receptors_dir/processed"
    mkdir -p "$processed_dir"
    mv "$pdb_path" "$processed_dir/$(basename "$pdb_path")" 2>/dev/null || true

    log "prep_receptor complete: $job_id"
}

run_dock_job() {
    local job_file="$1"
    local job_id receptor_dir library_dir

    job_id=$(jget       "$job_file" "['job_id']")
    receptor_dir=$(jget "$job_file" "['receptor_dir']")
    library_dir=$(jget  "$job_file" "['library_dir']")

    [[ -d "$receptor_dir" ]] || { log "ERROR: receptor_dir not found: $receptor_dir"; return 1; }
    [[ -d "$library_dir"  ]] || { log "ERROR: library_dir not found: $library_dir";  return 1; }

    # Extract options from job file
    local nrun use_conformers force max_failures
    nrun=$(jopt "$job_file" nrun)
    [[ -z "$nrun" || "$nrun" == "None" ]] && nrun=20
    use_conformers=$(jopt "$job_file" use_conformers)
    force=$(jopt "$job_file" force)
    max_failures=$(jopt "$job_file" max_failures)
    [[ -z "$max_failures" || "$max_failures" == "None" ]] && max_failures=5

    local extra_args="--nrun $nrun --max_failures $max_failures"
    [[ "$use_conformers" == "True" ]] && extra_args="$extra_args --use_conformers"
    [[ "$force"          == "True" ]] && extra_args="$extra_args --force"

    # Compounds filter
    local compounds
    compounds=$(python3 -c "
import json, sys
d = json.load(open('$job_file'))
c = d.get('options', {}).get('compounds')
if c:
    print('--compounds ' + ' '.join(str(x) for x in c))
" 2>/dev/null || echo "")
    [ -n "$compounds" ] && extra_args="$extra_args $compounds"

    log "Starting dock: $job_id"
    log "  Receptor: $receptor_dir"

    docker run --rm --gpus all $USER_ARG \
        --network none \
        -v "$receptor_dir":"$receptor_dir" \
        -v "$library_dir":"$library_dir" \
        "$DOCKING_IMAGE" dock_batch.py \
        --receptor_dir "$receptor_dir" \
        --library_dir  "$library_dir" \
        $extra_args \
        >> "$LOG_DIR/${job_id}.log" 2>&1

    log "dock complete: $job_id"
}


# ── Main loop ──────────────────────────────────────────────────────────────────
log "AutoDock job runner started (PID $$)"
log_startup_paths

while true; do
    for job_file in "$JOBS_DIR/pending/"*.json; do
        [[ -f "$job_file" ]] || continue

        job_name=$(basename "$job_file")
        running_file="$JOBS_DIR/running/$job_name"

        # Atomic move — prevents double-execution
        mv "$job_file" "$running_file" 2>/dev/null || continue

        log "Picked up: $job_name"

        # Stamp which host claimed the job and when it started, so the queue
        # view can show elapsed runtime and the machine the job is on (jobs may
        # run on either host sharing the queue).
        python3 - "$running_file" "$(hostname)" << 'PYEOF'
import json, sys
from datetime import datetime
d = json.load(open(sys.argv[1]))
d["host"]       = sys.argv[2]
d["started_at"] = datetime.now().isoformat()
open(sys.argv[1], "w").write(json.dumps(d, indent=2))
PYEOF

        job_type=$(jget "$running_file" "['type']")
        result="completed"

        # Exclusive lock - only one job runs at a time
        exec 200>/var/lock/autodock-job-runner.lock
        flock -x 200

        case "$job_type" in
            prep_ligands)  run_prep_ligands_job  "$running_file" || result="failed" ;;
            prep_receptor) run_prep_receptor_job "$running_file" || result="failed" ;;
            dock)          run_dock_job           "$running_file" || result="failed" ;;
            *) log "Unknown job type: $job_type"; result="failed" ;;
        esac

        # Release lock
        flock -u 200

        # Write completion timestamp
        python3 - "$running_file" "$result" << 'PYEOF'
import json, sys
from datetime import datetime
d = json.load(open(sys.argv[1]))
d["completed_at"] = datetime.now().isoformat()
d["result"]       = sys.argv[2]
open(sys.argv[1], "w").write(json.dumps(d, indent=2))
PYEOF

        mv "$running_file" "$JOBS_DIR/$result/$job_name"
        log "Job $job_name → $result"

        # Process one job per iteration then sleep
        break
    done
    sleep 5
done
