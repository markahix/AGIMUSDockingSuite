#!/usr/bin/env python3
"""
active_sites.py — Active site detection and box computation.

Given a PDB file and one or more HETATM residue names, identifies
distinct binding site instances (by spatial clustering), computes
search boxes, and merges overlapping boxes with reduced padding.
"""

import math
from pathlib import Path


# ── Padding reduction factor when boxes are merged ────────────────────────────
# Two boxes with 10Å padding each, when merged, get this reduced padding.
MERGED_PADDING_FACTOR = 0.5  # e.g. 10Å → 5Å


# ══════════════════════════════════════════════════════════════════════════════
# PDB parsing
# ══════════════════════════════════════════════════════════════════════════════

# Residues that are never treated as a ligand/active-site marker.
# Standard polymer residues plus solvent and common crystallisation ions.
_POLYMER_RESIDUES = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "HSD","HSE","HSP","HIE","HID","HIP","CYX","CYM","GLH","ASH",
    "LYN","ARN","SEC","PYL","MSE","ACE","NME","NMA","NHE",
    "DA","DC","DG","DT","DU","A","C","G","T","U",
    "RA","RC","RG","RU",
}
_SOLVENT_AND_IONS = {
    "HOH","WAT","SOL","H2O","TIP","TIP3","TIP4","DOD",
    "NA","CL","K","BR","IOD","CS","RB","LI","F",
    "SO4","PO4","NO3","ACT","EDO","GOL","PEG","MPD","DMS","TRS","EPE",
}


def parse_ligand_instances(pdb_path: Path,
                            resnames: list[str] | None = None) -> dict[str, list[dict]]:
    """
    Parse ligand residue instances from a PDB file.

    Scans BOTH ATOM and HETATM records: structures produced by modelling
    programs (MODELLER, etc.) frequently write bound ligands as ATOM.

    resnames: residue names to match. If None or empty, every residue that
              is not a standard polymer residue, solvent, or common ion is
              treated as a candidate ligand.

    Returns:
        {resname: [{"resname","chain","resseq","atoms":[(x,y,z),...],"centroid"}, ...]}
    """
    target = {r.strip().upper() for r in (resnames or []) if r.strip()}
    auto   = not target
    instances: dict[str, dict[tuple, list]] = {}

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            rname = line[17:20].strip().upper()
            if auto:
                if rname in _POLYMER_RESIDUES or rname in _SOLVENT_AND_IONS:
                    continue
            elif rname not in target:
                continue
            try:
                chain  = line[21].strip()
                resseq = line[22:26].strip()
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except (ValueError, IndexError):
                continue
            key = (rname, chain, resseq)
            instances.setdefault(rname, {})
            instances[rname].setdefault(key, []).append((x, y, z))

    result = {}
    for rname, keyed in instances.items():
        result[rname] = []
        for (rn, chain, resseq), atoms in keyed.items():
            result[rname].append({
                "resname": rn,
                "chain":   chain,
                "resseq":  resseq,
                "atoms":   atoms,
                "centroid": _centroid(atoms),
            })

    return result


def _centroid(atoms: list[tuple]) -> tuple[float, float, float]:
    n = len(atoms)
    return (sum(a[0] for a in atoms) / n,
            sum(a[1] for a in atoms) / n,
            sum(a[2] for a in atoms) / n)


# ══════════════════════════════════════════════════════════════════════════════
# Box computation
# ══════════════════════════════════════════════════════════════════════════════

def atoms_to_box(atoms: list[tuple[float, float, float]],
                 padding: float) -> dict:
    """Compute a search box from a list of atom coordinates + padding."""
    xs = [a[0] for a in atoms]
    ys = [a[1] for a in atoms]
    zs = [a[2] for a in atoms]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    cz = (min(zs) + max(zs)) / 2
    sx = round(max(xs) - min(xs) + 2 * padding, 1)
    sy = round(max(ys) - min(ys) + 2 * padding, 1)
    sz = round(max(zs) - min(zs) + 2 * padding, 1)
    return {"cx": cx, "cy": cy, "cz": cz,
            "sx": sx, "sy": sy, "sz": sz}


def boxes_overlap(a: dict, b: dict) -> bool:
    """True if two axis-aligned boxes overlap in 3D."""
    def overlap_1d(c1, s1, c2, s2):
        return abs(c1 - c2) < (s1 + s2) / 2
    return (overlap_1d(a["cx"], a["sx"], b["cx"], b["sx"]) and
            overlap_1d(a["cy"], a["sy"], b["cy"], b["sy"]) and
            overlap_1d(a["cz"], a["sz"], b["cz"], b["sz"]))


def merge_boxes(boxes: list[dict], padding: float) -> dict:
    """
    Merge multiple overlapping boxes into one, with reduced padding.
    Padding is halved for merged boxes to avoid enormous search spaces.
    """
    # Collect all atoms from all instances and rebuild with reduced padding
    # (We don't store atoms in boxes, so recompute from box corners)
    # Instead: take the union of all box extents
    min_x = min(b["cx"] - b["sx"]/2 for b in boxes)
    max_x = max(b["cx"] + b["sx"]/2 for b in boxes)
    min_y = min(b["cy"] - b["sy"]/2 for b in boxes)
    max_y = max(b["cy"] + b["sy"]/2 for b in boxes)
    min_z = min(b["cz"] - b["sz"]/2 for b in boxes)
    max_z = max(b["cz"] + b["sz"]/2 for b in boxes)

    # Reduced padding for merged boxes
    reduced_pad = padding * MERGED_PADDING_FACTOR

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    cz = (min_z + max_z) / 2
    sx = round(max_x - min_x + 2 * reduced_pad, 1)
    sy = round(max_y - min_y + 2 * reduced_pad, 1)
    sz = round(max_z - min_z + 2 * reduced_pad, 1)

    return {"cx": cx, "cy": cy, "cz": cz,
            "sx": sx, "sy": sy, "sz": sz,
            "merged": True, "n_merged": len(boxes),
            "reduced_padding": reduced_pad}


# ══════════════════════════════════════════════════════════════════════════════
# Main: detect active sites
# ══════════════════════════════════════════════════════════════════════════════

def detect_active_sites(pdb_path: Path,
                         resnames: list[str] | None = None,
                         padding: float = 10.0) -> list[dict]:
    """
    Detect distinct active sites from one or more HETATM residue names.

    Each unique instance of each residue name is treated as a candidate
    active site. Overlapping sites are merged into a single larger box
    with reduced padding.

    Returns list of active site dicts:
        {
          "label":    str,         # e.g. "AS1", "AS2", "AS1+AS2"
          "resnames": [str, ...],  # source residue name(s)
          "instances":[{chain,resseq,...},...],
          "box":      {cx,cy,cz,sx,sy,sz,...},
          "merged":   bool,
        }
    """
    instances_by_name = parse_ligand_instances(pdb_path, resnames)

    if not instances_by_name:
        return []

    # Build one candidate site per instance
    candidates = []
    for rname, instances in instances_by_name.items():
        for inst in instances:
            box = atoms_to_box(inst["atoms"], padding)
            candidates.append({
                "label":     f"{rname}_{inst['chain']}{inst['resseq']}",
                "resnames":  [rname],
                "instances": [inst],
                "box":       box,
                "merged":    False,
            })

    if len(candidates) <= 1:
        # Relabel single site
        if candidates:
            candidates[0]["label"] = "AS1"
        return candidates

    # Cluster overlapping candidates using union-find
    n = len(candidates)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if boxes_overlap(candidates[i]["box"], candidates[j]["box"]):
                union(i, j)

    # Group by cluster
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    # Build merged sites
    sites = []
    for cluster_indices in sorted(clusters.values(), key=lambda c: c[0]):
        members = [candidates[i] for i in cluster_indices]
        all_instances = [inst for m in members for inst in m["instances"]]
        all_resnames  = list(dict.fromkeys(
            rn for m in members for rn in m["resnames"]))

        if len(members) == 1:
            sites.append({
                "label":     "",          # filled in below
                "resnames":  all_resnames,
                "instances": all_instances,
                "box":       members[0]["box"],
                "merged":    False,
            })
        else:
            merged_box = merge_boxes([m["box"] for m in members], padding)
            sites.append({
                "label":     "",
                "resnames":  all_resnames,
                "instances": all_instances,
                "box":       merged_box,
                "merged":    True,
                "n_merged":  len(members),
                "reduced_padding": merged_box.get("reduced_padding", padding/2),
            })

    # Assign sequential labels
    for i, site in enumerate(sites):
        if len(sites) == 1:
            site["label"] = "AS1"
        else:
            site["label"] = f"AS{i+1}"

    return sites


# ══════════════════════════════════════════════════════════════════════════════
# Ensemble starting model selection
# ══════════════════════════════════════════════════════════════════════════════

def count_atoms_in_box(pdb_lines: list[str], box: dict) -> int:
    """
    Count ATOM records with Cα coordinates inside the search box.
    Used to select the starting model for ensemble screening
    (fewer atoms in box = less steric occlusion for initial ligand sampling).
    """
    cx, cy, cz = box["cx"], box["cy"], box["cz"]
    sx, sy, sz = box["sx"], box["sy"], box["sz"]
    count = 0
    for line in pdb_lines:
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            if (abs(x - cx) <= sx / 2 and
                abs(y - cy) <= sy / 2 and
                abs(z - cz) <= sz / 2):
                count += 1
        except (ValueError, IndexError):
            continue
    return count


def select_screening_model(ensemble_path: Path) -> int:
    """
    Select the best starting model for ensemble screening:
    the one with the fewest Cα atoms inside the search box.

    Returns the model_id (1-based) of the recommended starting model.
    """
    import json
    data = json.loads(ensemble_path.read_text())
    box  = data.get("box", {})
    if not box:
        return 1

    conformations = data.get("conformations", [])
    if not conformations:
        return 1

    best_model_id = conformations[0].get("model_id", 1)
    best_count    = None

    for conf in conformations:
        prep_dir = Path(conf["prep_dir"])
        fixed_pdb = prep_dir / "receptor_fixed.pdb"
        if not fixed_pdb.exists():
            continue
        lines = fixed_pdb.read_text().splitlines()
        n = count_atoms_in_box(lines, box)
        if best_count is None or n < best_count:
            best_count    = n
            best_model_id = conf.get("model_id", 1)

    return best_model_id


# Backwards-compatible alias
parse_hetatm_instances = parse_ligand_instances
