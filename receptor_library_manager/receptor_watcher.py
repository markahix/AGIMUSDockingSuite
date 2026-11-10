#!/usr/bin/env python3
"""
receptor_watcher.py
-------------------
Continuously watches the receptors/inbox directory for new PDB files
and processes them automatically.

Usage:
    python3 receptor_watcher.py \
        --receptors_dir /receptors \
        --library_dir /library \
        --center_x -23.0 --center_y 14.2 --center_z 60.3 \
        --autobox \
        --interval 60
"""

import argparse
import hashlib
import shutil
import signal
import sys
import time
from pathlib import Path

_running = True
_seen    = {}  # filename → hash, to avoid reprocessing unchanged files


def handle_signal(sig, frame):
    global _running
    print(f"\n[watcher] Signal {sig} received — shutting down")
    _running = False


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def scan_inbox(inbox: Path, receptors_dir: Path, library_dir: Path,
               box_args: list[str], autobox: bool, padding: float,
               rmsd_threshold: float, ph: float, spacing: float) -> tuple[int, int]:
    """Check inbox for new PDB files and process them."""
    n_proc = n_fail = 0

    for pdb_path in sorted(inbox.glob("*.pdb")):
        h = file_hash(pdb_path)
        if _seen.get(pdb_path.name) == h:
            continue  # already processed this exact file

        print(f"[watcher] New receptor: {pdb_path.name}", flush=True)
        _seen[pdb_path.name] = h

        # Build prep_receptor.py command
        cmd = [
            sys.executable, "/app/prep_receptor.py",
            "--pdb",           str(pdb_path),
            "--receptors_dir", str(receptors_dir),
            "--library_dir",   str(library_dir),
            "--spacing",       str(spacing),
            "--ph",            str(ph),
            "--rmsd_threshold", str(rmsd_threshold),
        ]
        if autobox:
            cmd += ["--autobox", "--padding", str(padding)]
        cmd += box_args

        import subprocess
        result = subprocess.run(cmd)

        if result.returncode == 0:
            print(f"[watcher] Processed: {pdb_path.name}", flush=True)
            # Move processed file out of inbox
            done_dir = inbox.parent / "processed"
            done_dir.mkdir(exist_ok=True)
            shutil.move(str(pdb_path), done_dir / pdb_path.name)
            n_proc += 1
        else:
            print(f"[watcher] FAILED: {pdb_path.name}", flush=True)
            fail_dir = inbox.parent / "failed"
            fail_dir.mkdir(exist_ok=True)
            shutil.move(str(pdb_path), fail_dir / pdb_path.name)
            n_fail += 1

    return n_proc, n_fail


def main():
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT,  handle_signal)

    p = argparse.ArgumentParser(
        description="Watch receptor inbox and process new PDB files."
    )
    p.add_argument("--receptors_dir", default="/receptors")
    p.add_argument("--library_dir",   default="/library")
    p.add_argument("--interval",      type=int, default=60)
    p.add_argument("--center_x",      type=float, required=True)
    p.add_argument("--center_y",      type=float, required=True)
    p.add_argument("--center_z",      type=float, required=True)
    p.add_argument("--size_x",        type=float, default=None)
    p.add_argument("--size_y",        type=float, default=None)
    p.add_argument("--size_z",        type=float, default=None)
    p.add_argument("--autobox",       action="store_true")
    p.add_argument("--padding",       type=float, default=10.0)
    p.add_argument("--rmsd_threshold",type=float, default=0.5)
    p.add_argument("--ph",            type=float, default=7.4)
    p.add_argument("--spacing",       type=float, default=0.375)
    args = p.parse_args()

    receptors_dir = Path(args.receptors_dir)
    inbox         = receptors_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)

    # Build box args to pass through to prep_receptor.py
    box_args = [
        "--center_x", str(args.center_x),
        "--center_y", str(args.center_y),
        "--center_z", str(args.center_z),
    ]
    if args.size_x: box_args += ["--size_x", str(args.size_x)]
    if args.size_y: box_args += ["--size_y", str(args.size_y)]
    if args.size_z: box_args += ["--size_z", str(args.size_z)]

    print(f"[watcher] Watching {inbox} every {args.interval}s", flush=True)

    while _running:
        n_proc, n_fail = scan_inbox(
            inbox, receptors_dir, Path(args.library_dir),
            box_args, args.autobox, args.padding,
            args.rmsd_threshold, args.ph, args.spacing
        )
        if n_proc or n_fail:
            print(f"[watcher] {n_proc} processed, {n_fail} failed", flush=True)

        for _ in range(args.interval * 2):
            if not _running: break
            time.sleep(0.5)

    print("[watcher] Stopped")


if __name__ == "__main__":
    main()
