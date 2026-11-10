#!/usr/bin/env python3
"""
query_library.py
----------------
Query and filter the ligand library by properties.

Usage:
    # List all ready ligands
    python3 query_library.py --library_dir /library

    # Filter by size
    python3 query_library.py --library_dir /library \
        --max_atoms 100 --max_rot 30

    # Filter by element
    python3 query_library.py --library_dir /library --elements N,S

    # Show summary only
    python3 query_library.py --library_dir /library --summary

    # Output as JSON
    python3 query_library.py --library_dir /library --format json
"""

import argparse
import json
import sys
from pathlib import Path

from library_state import LibraryState


def main():
    p = argparse.ArgumentParser(description="Query the ligand library.")
    p.add_argument("--library_dir", required=True)
    p.add_argument("--status",     default="ready",
                   help="Filter by status (default: ready)")
    p.add_argument("--min_atoms",  type=int, default=None)
    p.add_argument("--max_atoms",  type=int, default=None)
    p.add_argument("--min_rot",    type=int, default=None)
    p.add_argument("--max_rot",    type=int, default=None)
    p.add_argument("--elements",   type=str, default=None,
                   help="Required elements, comma-separated (e.g. N,S)")
    p.add_argument("--summary",    action="store_true",
                   help="Print status summary only")
    p.add_argument("--format",     choices=["table", "names", "json"],
                   default="table")
    args = p.parse_args()

    state = LibraryState(Path(args.library_dir))

    if args.summary:
        s = state.summary()
        total = sum(s.values())
        print(f"Library: {Path(args.library_dir)}")
        for status, n in sorted(s.items()):
            print(f"  {status:<12} {n}")
        print(f"  {'total':<12} {total}")
        return

    rows = state.get_ready() if args.status == "ready" else []

    # Apply filters
    req_elems = set(args.elements.split(",")) if args.elements else None

    filtered = []
    for row in rows:
        if args.min_atoms and row["n_heavy_atoms"] < args.min_atoms: continue
        if args.max_atoms and row["n_heavy_atoms"] > args.max_atoms: continue
        if args.min_rot   and row["n_rot_bonds"]   < args.min_rot:   continue
        if args.max_rot   and row["n_rot_bonds"]   > args.max_rot:   continue
        if req_elems:
            elems = set(json.loads(row["elements_json"] or "[]"))
            if not req_elems.issubset(elems): continue
        filtered.append(row)

    if args.format == "names":
        for row in filtered:
            print(row["name"])

    elif args.format == "json":
        out = {r["name"]: json.loads(r["bbox_json"] or "{}") for r in filtered}
        print(json.dumps(out, indent=2))

    else:  # table
        if not filtered:
            print("No ligands match the criteria.")
            return
        header = f"{'Name':<30} {'Formula':<20} {'Heavy':>5} {'Rot':>4} {'Confs':>5} {'Mode':<12}"
        print(header)
        print("-" * len(header))
        for row in filtered:
            mode = "flex-fallback" if row["use_flex_fallback"] else (
                   f"conformers({row['n_conformers']})" if row["n_conformers"] > 0
                   else "flex")
            print(f"{row['name']:<30} {row['formula']:<20} "
                  f"{row['n_heavy_atoms']:>5} {row['n_rot_bonds']:>4} "
                  f"{row['n_conformers']:>5} {mode:<12}")
        print(f"\n{len(filtered)} ligand(s) shown")


if __name__ == "__main__":
    main()
