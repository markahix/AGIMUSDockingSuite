"""
receptor_identity.py
--------------------
Receptor identity matching by Cα amino acid sequence.

Extracts the amino acid sequence from Cα ATOM records in PDB files,
then compares against sequences stored in existing receptor directories.

Match outcomes:
  new         — no existing receptor matches
  exact       — sequence matches an existing receptor exactly
  ambiguous   — partial match (e.g. terminal truncation/extension)

Ambiguous receptors are always treated as new (per design decision).
"""

import json
import sqlite3
from pathlib import Path


# Standard amino acid 3-letter to 1-letter code
_AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common variants
    "HSD": "H", "HSE": "H", "HSP": "H",
    "HIE": "H", "HID": "H", "HIP": "H",
    "CYX": "C", "CYM": "C",
    "GLH": "E", "ASH": "D",
    "LYN": "K", "ARN": "R",
    "SEC": "U", "PYL": "O",
    "MSE": "M",  # selenomethionine
}


def extract_sequence(pdb_lines: list[str]) -> str:
    """
    Extract amino acid sequence from Cα ATOM records.
    Returns fingerprint string: "A:MGSS...,B:DL..."
    Chains are sorted alphabetically.
    Missing residues (gaps in resseq) are NOT filled — sequence is
    exactly what is present in the PDB.
    """
    chains = {}
    seen   = {}  # (chain, resseq) → already added

    for line in pdb_lines:
        if not line.startswith("ATOM"):
            continue
        aname = line[12:16].strip()
        if aname != "CA":
            continue
        try:
            chain   = line[21].strip() or "A"
            resname = line[17:20].strip()
            resseq  = int(line[22:26])
        except (ValueError, IndexError):
            continue

        key = (chain, resseq)
        if key in seen:
            continue
        seen[key] = True

        aa = _AA3TO1.get(resname, "X")
        chains.setdefault(chain, []).append((resseq, aa))

    # Build sorted sequence per chain
    parts = []
    for chain in sorted(chains.keys()):
        residues = sorted(chains[chain], key=lambda r: r[0])
        seq      = "".join(aa for _, aa in residues)
        parts.append(f"{chain}:{seq}")

    return ",".join(parts)


def match_receptor(sequence: str, receptors_dir: Path) -> tuple[str, str, str]:
    """
    Compare sequence against all existing receptors.

    Returns (match_type, receptor_name, notes) where match_type is one of:
      'new'       — no match found
      'exact'     — exact sequence match
      'ambiguous' — partial match (treated as new per policy)
    """
    if not receptors_dir.exists():
        return "new", "", "No receptors directory"

    matches = []

    for seq_file in receptors_dir.glob("*/receptor_sequence.txt"):
        receptor_name = seq_file.parent.name
        try:
            existing_seq = seq_file.read_text().strip()
        except Exception:
            continue

        if existing_seq == sequence:
            matches.append(("exact", receptor_name))
            continue

        # Check for partial match: same chains, one is subset of other
        if _is_partial_match(sequence, existing_seq):
            matches.append(("ambiguous", receptor_name))

    if not matches:
        return "new", "", "No matching receptor found"

    # Prefer exact matches
    exact = [m for m in matches if m[0] == "exact"]
    if exact:
        name = exact[0][1]
        return "exact", name, f"Matches existing receptor '{name}'"

    # Ambiguous — treat as new
    names = [m[1] for m in matches]
    return ("ambiguous", "",
            f"Partial sequence match with: {', '.join(names)} — treating as new")


def _is_partial_match(seq_a: str, seq_b: str) -> bool:
    """
    True if seq_a and seq_b share the same chains with sequences
    that are substrings of each other (terminal truncation/extension).
    """
    def parse(seq):
        parts = {}
        for part in seq.split(","):
            if ":" not in part:
                continue
            chain, s = part.split(":", 1)
            parts[chain] = s
        return parts

    a = parse(seq_a)
    b = parse(seq_b)

    if set(a.keys()) != set(b.keys()):
        return False  # different chains

    for chain in a:
        sa, sb = a[chain], b[chain]
        # Check if one is a substring of the other at either end
        if not (sa in sb or sb in sa or
                sa.startswith(sb[:10]) or sb.startswith(sa[:10])):
            return False

    return True


def suggest_name(pdb_path: Path, receptors_dir: Path) -> str:
    """
    Suggest a directory name for a new receptor.
    Uses the PDB stem, with a numeric suffix if already taken.
    """
    base = pdb_path.stem
    candidate = base
    n = 1
    while (receptors_dir / candidate).exists():
        n += 1
        candidate = f"{base}_{n}"
    return candidate
