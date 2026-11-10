#!/usr/bin/env python3
"""
library_watcher.py
------------------
Continuously watches a ligand directory for new or changed PDB files
and processes them into the library automatically.

Uses polling (no inotify dependency) — checks every --interval seconds.
Designed to run as a long-lived container service.

Usage:
    python3 library_watcher.py \
        --ligand_dir /ligands \
        --library_dir /library \
        --interval 30
"""

import argparse
import signal
import sys
import time
from pathlib import Path

from library_state import LibraryState
from prep_ligand import file_hash, resolve_name, process_ligand

_running = True


def handle_signal(sig, frame):
    global _running
    print(f"\n[watcher] Signal {sig} received — shutting down cleanly")
    _running = False


def scan_once(ligand_dir: Path, library_dir: Path,
              state: LibraryState, pattern: str,
              conformers: int | None) -> tuple[int, int]:
    """
    Scan ligand_dir for new/changed PDBs and process them.
    Returns (n_processed, n_failed).
    """
    n_proc = n_fail = 0
    for pdb_path in sorted(ligand_dir.glob(pattern)):
        name = resolve_name(pdb_path, library_dir)
        h    = file_hash(pdb_path)
        updated = state.register(name, h)
        if not updated:
            continue  # unchanged

        print(f"[watcher] Detected: {pdb_path.name} → {name}", flush=True)
        ok = process_ligand(pdb_path, library_dir, state,
                            n_confs_override=conformers, verbose=True)
        if ok:
            n_proc += 1
        else:
            n_fail += 1
    return n_proc, n_fail


def main():
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT,  handle_signal)

    p = argparse.ArgumentParser(
        description="Watch ligand directory and process new PDBs automatically."
    )
    p.add_argument("--ligand_dir",  required=True)
    p.add_argument("--library_dir", required=True)
    p.add_argument("--interval",    type=int, default=30,
                   help="Polling interval in seconds (default: 30)")
    p.add_argument("--pattern",     default="*.pdb")
    p.add_argument("--conformers",  type=int, default=None)
    args = p.parse_args()

    ligand_dir  = Path(args.ligand_dir)
    library_dir = Path(args.library_dir)

    if not ligand_dir.exists():
        print(f"ERROR: ligand_dir not found: {ligand_dir}", file=sys.stderr)
        sys.exit(1)

    library_dir.mkdir(parents=True, exist_ok=True)
    state = LibraryState(library_dir)

    print(f"[watcher] Starting — watching {ligand_dir} every {args.interval}s")
    summary = state.summary()
    print(f"[watcher] Library state: {summary}", flush=True)

    while _running:
        n_proc, n_fail = scan_once(
            ligand_dir, library_dir, state, args.pattern, args.conformers)
        if n_proc or n_fail:
            print(f"[watcher] Cycle complete: "
                  f"{n_proc} processed, {n_fail} failed", flush=True)
        # Sleep in small increments to respond to signals promptly
        for _ in range(args.interval * 2):
            if not _running:
                break
            time.sleep(0.5)

    print("[watcher] Stopped cleanly")


if __name__ == "__main__":
    main()
