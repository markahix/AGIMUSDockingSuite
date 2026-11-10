#!/usr/bin/env python3
"""
prep_ligand.py
--------------
Process one or more PDB ligand files into the library.
Reads from --ligand_dir, writes to --library_dir.
State tracked via SQLite (library_state.db).

Usage:
    # Process all new/changed ligands
    python3 prep_ligand.py --ligand_dir /ligands --library_dir /library

    # Force reprocess specific ligands
    python3 prep_ligand.py --ligand_dir /ligands --library_dir /library \
        --refresh Ligand_01 Ligand_02

    # Force reprocess all
    python3 prep_ligand.py --ligand_dir /ligands --library_dir /library \
        --refresh_all

    # Override conformer count
    python3 prep_ligand.py --ligand_dir /ligands --library_dir /library \
        --conformers 50

    # Disable conformer generation
    python3 prep_ligand.py --ligand_dir /ligands --library_dir /library \
        --conformers 0
"""

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

from library_state import LibraryState

# ── VDW radii for clash detection ─────────────────────────────────────────────
_VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "P": 1.80, "F": 1.47, "Cl": 1.75, "Br": 1.85, "I": 1.98,
}
_CLASH_SCALE = 0.55


# ══════════════════════════════════════════════════════════════════════════════
# File utilities
# ══════════════════════════════════════════════════════════════════════════════

def file_hash(path: Path, n: int = 12) -> str:
    """MD5 hash of file contents (first n hex chars)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def resolve_name(pdb_path: Path, library_dir: Path) -> str:
    """
    Determine canonical ligand name from PDB filename.
    Handles collisions by appending hash suffix.
    """
    stem = pdb_path.stem
    comp_dir = library_dir / stem
    if not comp_dir.exists():
        return stem
    # Check if existing entry is the same file
    existing_link = comp_dir / f"{stem}.pdb"
    if existing_link.exists():
        try:
            if existing_link.resolve() == pdb_path.resolve():
                return stem
        except Exception:
            pass
    # Collision — append hash
    h = file_hash(pdb_path, 6)
    return f"{stem}_{h}"


# ══════════════════════════════════════════════════════════════════════════════
# Atom parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_atoms(pdb_path: Path) -> list[dict]:
    """Parse ATOM/HETATM records from PDB. Returns list of atom dicts."""
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                serial  = int(line[6:11])
                aname   = line[12:16].strip()
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                element = line[76:78].strip() if len(line) > 76 else ""
                if not element:
                    raw = "".join(c for c in aname if c.isalpha())
                    element = (raw[0].upper() + raw[1:].lower()
                               if len(raw) > 1 else raw.upper())
                atoms.append({
                    "serial": serial, "aname": aname,
                    "x": x, "y": y, "z": z, "element": element
                })
            except (ValueError, IndexError):
                continue
    return atoms


def build_bonds(atoms: list[dict]) -> list[list[int]]:
    """Build adjacency list from proximity (< 2.0 Å)."""
    import numpy as np
    n      = len(atoms)
    coords = [[a["x"], a["y"], a["z"]] for a in atoms]
    bonds  = [[] for _ in range(n)]
    for i in range(n):
        xi, yi, zi = coords[i]
        for j in range(i+1, n):
            dx = xi - coords[j][0]
            dy = yi - coords[j][1]
            dz = zi - coords[j][2]
            if dx*dx + dy*dy + dz*dz < 4.0:  # 2.0² = 4.0
                bonds[i].append(j)
                bonds[j].append(i)
    return bonds


# ══════════════════════════════════════════════════════════════════════════════
# Aromaticity detection
# ══════════════════════════════════════════════════════════════════════════════

def get_aromaticity(atoms: list[dict], bonds: list[list[int]]) -> dict[int, bool]:
    """
    Detect aromatic atoms via geometry: find planar 5/6-membered rings.
    Returns {atom_index: is_aromatic}.
    """
    import numpy as np

    n      = len(atoms)
    coords = [([a["x"], a["y"], a["z"]]) for a in atoms]
    elems  = [a["element"] for a in atoms]

    def find_rings(start, max_size=6):
        rings = []
        def dfs(path, visited):
            if len(path) > max_size:
                return
            cur = path[-1]
            for nb in bonds[cur]:
                if nb == path[0] and len(path) >= 5:
                    rings.append(tuple(path))
                elif nb not in visited:
                    dfs(path + [nb], visited | {nb})
        dfs([start], {start})
        return rings

    all_rings = set()
    for i in range(n):
        if elems[i] in ("C", "N", "O", "S"):
            for ring in find_rings(i):
                all_rings.add(tuple(sorted(ring)))

    def is_planar(ring_indices):
        heavy = [i for i in ring_indices if elems[i] != "H"]
        if len(heavy) < 5:
            return False
        pts = np.array([coords[i] for i in heavy])
        cen = pts.mean(axis=0)
        try:
            _, _, vh = np.linalg.svd(pts - cen)
            devs = [abs(float(np.dot(np.array(coords[i]) - cen, vh[-1])))
                    for i in heavy]
            return max(devs) < 0.15
        except Exception:
            return False

    def is_sp2_ring(ring_indices):
        for i in ring_indices:
            if elems[i] == "H":
                continue
            if sum(1 for nb in bonds[i] if elems[nb] != "H") > 3:
                return False
        return True

    aromatic = set()
    for ring in all_rings:
        heavy = [i for i in ring if elems[i] != "H"]
        if len(heavy) >= 5 and is_planar(ring) and is_sp2_ring(ring):
            aromatic.update(heavy)

    return {i: (i in aromatic) for i in range(n)}


# ══════════════════════════════════════════════════════════════════════════════
# Rotatable bond detection
# ══════════════════════════════════════════════════════════════════════════════

def get_rotatable_bonds(atoms: list[dict], bonds: list[list[int]],
                        aromaticity: dict[int, bool]) -> list[tuple[int, int]]:
    """
    Identify rotatable bonds: sp3-sp3, non-ring, non-terminal.
    Excludes sp2-sp2 bonds (aromatic, amide, vinyl).
    """
    import numpy as np

    n      = len(atoms)
    elems  = [a["element"] for a in atoms]
    coords = [[a["x"], a["y"], a["z"]] for a in atoms]

    def in_ring(i, j):
        visited, queue = {i}, [nb for nb in bonds[i] if nb != j]
        while queue:
            cur = queue.pop(0)
            if cur == j: return True
            if cur in visited: continue
            visited.add(cur)
            queue.extend(nb for nb in bonds[cur] if nb not in visited)
        return False

    def is_sp2(i):
        if elems[i] == "H": return False
        if aromaticity.get(i, False): return True
        heavy_nb = [nb for nb in bonds[i] if elems[nb] != "H"]
        if len(heavy_nb) != 3: return False
        pts = np.array([coords[i]] + [coords[nb] for nb in heavy_nb])
        cen = pts.mean(axis=0)
        try:
            _, _, vh = np.linalg.svd(pts - cen)
            return max(abs(float(np.dot(p - cen, vh[-1]))) for p in pts) < 0.15
        except Exception:
            return False

    rot_bonds = []
    for i in range(n):
        if elems[i] == "H": continue
        for j in bonds[i]:
            if j <= i or elems[j] == "H": continue
            if sum(1 for nb in bonds[i] if elems[nb] != "H") < 2: continue
            if sum(1 for nb in bonds[j] if elems[nb] != "H") < 2: continue
            if in_ring(i, j): continue
            if is_sp2(i) and is_sp2(j): continue
            rot_bonds.append((i, j))
    return rot_bonds


# ══════════════════════════════════════════════════════════════════════════════
# Gasteiger charges (via RDKit)
# ══════════════════════════════════════════════════════════════════════════════

def get_charges(pdb_path: Path) -> dict[int, float]:
    """Compute Gasteiger charges via RDKit. Returns {serial: charge}."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=True)
        if mol is None:
            return {}
        AllChem.ComputeGasteigerCharges(mol)
        out = {}
        for atom in mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            serial = info.GetSerialNumber() if info else atom.GetIdx() + 1
            charge = atom.GetDoubleProp("_GasteigerCharge")
            if not math.isfinite(charge):
                charge = 0.0
            out[serial] = round(charge, 4)
        return out
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# AD4 atom type assignment
# ══════════════════════════════════════════════════════════════════════════════

def ad4_type(element: str, aname: str, neighbor_elems: list[str],
             is_aromatic: bool = False) -> str:
    """Assign AutoDock4 atom type."""
    e = element.strip().capitalize()
    if e == "C":  return "A" if is_aromatic else "C"
    if e == "N":  return "N" if any(nb == "H" for nb in neighbor_elems) else "NA"
    if e == "O":  return "OA"
    if e == "S":  return "SA" if any(nb == "H" for nb in neighbor_elems) else "S"
    if e == "H":  return "HD" if any(nb in ("N","O","S") for nb in neighbor_elems) else "H"
    return {"P":"P","F":"F","CL":"Cl","BR":"Br","I":"I",
            "ZN":"Zn","FE":"Fe","MG":"Mg","CA":"Ca","MN":"Mn"}.get(e.upper(), e)


# ══════════════════════════════════════════════════════════════════════════════
# PDBQT writer
# ══════════════════════════════════════════════════════════════════════════════

def write_flexible_pdbqt(atoms: list[dict], bonds: list[list[int]],
                          aromaticity: dict[int, bool],
                          rot_bonds: list[tuple[int, int]],
                          charges: dict[int, float],
                          out_path: Path,
                          coords_override: list | None = None) -> bool:
    """
    Write a flexible PDBQT with ROOT/BRANCH/ENDBRANCH/TORSDOF structure.
    coords_override: if provided, use these (x,y,z) tuples instead of atom coords.
    Returns True on success.
    """
    n      = len(atoms)
    elems  = [a["element"] for a in atoms]
    coords = coords_override or [(a["x"], a["y"], a["z"]) for a in atoms]

    rot_set  = {frozenset(b) for b in rot_bonds}
    n_tor    = len(rot_bonds)

    # Root: most-connected heavy atom
    heavy  = [i for i in range(n) if elems[i] != "H"]
    if not heavy:
        return False
    root   = max(heavy, key=lambda i: sum(1 for nb in bonds[i] if elems[nb] != "H"))

    serial_out = [0] * n
    out_serial  = [1]

    def assign(i):
        serial_out[i] = out_serial[0]
        out_serial[0] += 1

    def atom_line(i):
        x, y, z = coords[i]
        ch  = charges.get(atoms[i]["serial"], 0.0)
        nb  = [elems[k] for k in bonds[i]]
        typ = ad4_type(elems[i], atoms[i]["aname"], nb, aromaticity.get(i, False))
        return (f"ATOM  {serial_out[i]:5d} {atoms[i]['aname']:<4s} UNL A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    {ch:+6.3f} {typ:<2s}\n")

    def collect_fragment(start, seen):
        frag, pivots = [], []
        queue, local = [start], set()
        while queue:
            i = queue.pop(0)
            if i in seen or i in local: continue
            local.add(i); seen.add(i); assign(i); frag.append(i)
            for nb in bonds[i]:
                if nb in seen or nb in local: continue
                if frozenset((i, nb)) in rot_set:
                    pivots.append((i, nb))
                else:
                    queue.append(nb)
        return frag, pivots

    def build_branch(piv, ent, seen):
        lines, work = [], [("enter", piv, ent)]
        while work:
            task = work.pop()
            if task[0] == "exit":
                _, p, e = task
                lines.append(f"ENDBRANCH {serial_out[p]:3d} {serial_out[e]:3d}\n")
                continue
            _, p, e = task
            if e in seen: continue
            assign(e); seen.add(e)
            lines.append(f"BRANCH {serial_out[p]:3d} {serial_out[e]:3d}\n")
            lines.append(atom_line(e))
            frag, sub = collect_from(e, seen, p)
            lines.extend(atom_line(i) for i in frag)
            work.append(("exit", p, e))
            for sp, se in reversed(sub):
                work.append(("enter", sp, se))
        return lines

    def collect_from(start, seen, came_from):
        frag, pivots = [], []
        queue, local = [], set()
        for nb in bonds[start]:
            if nb == came_from or nb in seen: continue
            if frozenset((start, nb)) in rot_set:
                pivots.append((start, nb))
            else:
                queue.append(nb)
        while queue:
            i = queue.pop(0)
            if i in seen or i in local: continue
            local.add(i); seen.add(i); assign(i); frag.append(i)
            for nb in bonds[i]:
                if nb in seen or nb in local: continue
                if frozenset((i, nb)) in rot_set:
                    pivots.append((i, nb))
                else:
                    queue.append(nb)
        return frag, pivots

    seen = set()
    root_frag, root_pivots = collect_fragment(root, seen)

    out_lines = [f"REMARK {n_tor} active torsions\n", "ROOT\n"]
    out_lines.extend(atom_line(i) for i in root_frag)
    out_lines.append("ENDROOT\n")
    for pv, en in root_pivots:
        out_lines.extend(build_branch(pv, en, seen))
    out_lines.append(f"TORSDOF {n_tor}\n")

    out_path.write_text("".join(out_lines))
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Properties
# ══════════════════════════════════════════════════════════════════════════════

def compute_properties(pdb_path: Path, atoms: list[dict]) -> dict:
    """Compute ligand properties: formula, bbox, elements, SMILES."""
    from collections import Counter
    elems    = [a["element"] for a in atoms]
    n_heavy  = sum(1 for e in elems if e != "H")
    n_H      = sum(1 for e in elems if e == "H")

    # Hill formula
    counts = Counter(elems)
    formula = ""
    for e in ["C", "H"] + sorted(e for e in counts if e not in ("C", "H")):
        if counts.get(e, 0):
            formula += e + (str(counts[e]) if counts[e] > 1 else "")

    # Bounding box
    xs = [a["x"] for a in atoms]
    ys = [a["y"] for a in atoms]
    zs = [a["z"] for a in atoms]
    bbox = {
        "min_x": round(min(xs), 3), "max_x": round(max(xs), 3),
        "min_y": round(min(ys), 3), "max_y": round(max(ys), 3),
        "min_z": round(min(zs), 3), "max_z": round(max(zs), 3),
        "cx": round(sum(xs)/len(xs), 3),
        "cy": round(sum(ys)/len(ys), 3),
        "cz": round(sum(zs)/len(zs), 3),
        "dx": round(max(xs) - min(xs), 3),
        "dy": round(max(ys) - min(ys), 3),
        "dz": round(max(zs) - min(zs), 3),
    }

    # Unique heavy elements
    elements = sorted(set(e for e in elems if e != "H"))

    # SMILES via RDKit
    smiles = None
    smiles_source = "none"
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=True)
        if mol:
            smiles = Chem.MolToSmiles(mol)
            smiles_source = "computed"
    except Exception:
        pass

    # Check for user-provided .smi file
    smi_path = pdb_path.with_suffix(".smi")
    if smi_path.exists():
        smiles = smi_path.read_text().strip().split()[0]
        smiles_source = "user"

    return {
        "formula":      formula,
        "n_heavy_atoms": n_heavy,
        "n_hydrogens":  n_H,
        "bbox":         bbox,
        "elements":     elements,
        "smiles":       smiles,
        "smiles_source": smiles_source,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Conformer generation (Z-matrix / NeRF)
# ══════════════════════════════════════════════════════════════════════════════

def _nerf(a, b, c, bond_len, bond_angle, dihedral):
    import numpy as np
    bc      = c - b
    bc_norm = bc / np.linalg.norm(bc)
    ab      = b - a
    n_vec   = np.cross(ab, bc)
    if np.linalg.norm(n_vec) < 1e-10:
        n_vec = np.array([0.0, 0.0, 1.0])
    n_vec = n_vec / np.linalg.norm(n_vec)
    m     = np.cross(n_vec, bc_norm)
    d_loc = bond_len * np.array([
        -math.cos(bond_angle),
         math.sin(bond_angle) * math.cos(dihedral),
         math.sin(bond_angle) * math.sin(dihedral),
    ])
    return c + d_loc[0]*bc_norm + d_loc[1]*m + d_loc[2]*n_vec


def _has_clash(positions, elems, excluded):
    import numpy as np
    n = len(positions)
    for i in range(n):
        ri = _VDW_RADII.get(elems[i], 1.70)
        for j in range(i+1, n):
            if (i, j) in excluded: continue
            rj = _VDW_RADII.get(elems[j], 1.70)
            d  = float(np.linalg.norm(positions[i] - positions[j]))
            if d < _CLASH_SCALE * (ri + rj):
                return True
    return False


def default_conformer_count(n_heavy: int) -> int:
    if n_heavy < 50:  return 50
    if n_heavy < 100: return 100
    return 200


def generate_conformers(pdb_path: Path, atoms: list[dict],
                        bonds: list[list[int]],
                        aromaticity: dict[int, bool],
                        rot_bonds: list[tuple[int, int]],
                        charges: dict[int, float],
                        conf_dir: Path,
                        n_confs: int,
                        rmsd_threshold: float = 0.5) -> tuple[int, str]:
    """
    Generate diverse conformers using Z-matrix / NeRF internal coordinates.
    Returns (n_generated, error_or_empty).
    """
    import numpy as np

    n      = len(atoms)
    elems  = [a["element"] for a in atoms]
    coords = np.array([[a["x"], a["y"], a["z"]] for a in atoms])

    # Excluded pairs for clash detection (1-2 and 1-3)
    excluded = set()
    for i in range(n):
        for j in bonds[i]:
            excluded.add((min(i,j), max(i,j)))
            for k in bonds[j]:
                if k != i:
                    excluded.add((min(i,k), max(i,k)))

    if not rot_bonds:
        # Rigid molecule — write original as only conformer
        conf_dir.mkdir(parents=True, exist_ok=True)
        write_flexible_pdbqt(atoms, bonds, aromaticity, rot_bonds,
                             charges, conf_dir / "conf_0001.pdbqt")
        return 1, ""

    rot_set = {(min(i,j), max(i,j)) for i,j in rot_bonds}

    # Build spanning tree (BFS from most-connected heavy atom)
    heavy  = [i for i in range(n) if elems[i] != "H"]
    root   = max(heavy, key=lambda i: sum(1 for nb in bonds[i] if elems[nb] != "H"))

    parent  = [-1] * n
    order   = []
    visited = {root}
    queue   = [root]
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for nb in bonds[cur]:
            if nb not in visited:
                visited.add(nb)
                parent[nb] = cur
                queue.append(nb)

    # Build Z-matrix
    def calc_dihedral_angle(a, b, c, d):
        b1 = b-a; b2 = c-b; b3 = d-c
        n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
        nn1, nn2 = np.linalg.norm(n1), np.linalg.norm(n2)
        if nn1 < 1e-10 or nn2 < 1e-10: return 0.0
        n1 /= nn1; n2 /= nn2
        m1 = np.cross(n1, b2/np.linalg.norm(b2))
        return math.atan2(float(np.dot(m1,n2)), float(np.dot(n1,n2)))

    zmat = {}
    for idx in order:
        p = parent[idx]
        if p < 0:
            zmat[idx] = None; continue
        gp = parent[p]
        if gp < 0:
            zmat[idx] = (float(np.linalg.norm(coords[idx]-coords[p])),
                         None, None, False); continue
        ggp = parent[gp]
        bl  = float(np.linalg.norm(coords[idx]-coords[p]))
        ba  = math.acos(max(-1.0, min(1.0, float(np.dot(
            (coords[gp]-coords[p]) / np.linalg.norm(coords[gp]-coords[p]),
            (coords[idx]-coords[p]) / bl)))))
        if ggp < 0:
            zmat[idx] = (bl, ba, 0.0, False); continue
        dih    = calc_dihedral_angle(coords[ggp], coords[gp], coords[p], coords[idx])
        is_rot = (min(idx,p), max(idx,p)) in rot_set
        zmat[idx] = (bl, ba, dih, is_rot)

    def reconstruct(overrides):
        pos = np.zeros((n, 3))
        for idx in order:
            entry = zmat.get(idx)
            if entry is None:
                pos[idx] = coords[idx].copy(); continue
            bl, ba, dih, is_rot = entry
            p   = parent[idx]
            gp  = parent[p]  if p  >= 0 else -1
            ggp = parent[gp] if gp >= 0 else -1
            if ba is None or ggp < 0:
                pos[idx] = coords[idx].copy(); continue
            dih_use = overrides.get(idx, dih) if is_rot else dih
            if dih_use is None: dih_use = 0.0
            pos[idx] = _nerf(pos[ggp], pos[gp], pos[p], bl, ba, dih_use)
        return pos

    rot_indices = [idx for idx in order
                   if zmat.get(idx) and zmat[idx][3]]

    conf_dir.mkdir(parents=True, exist_ok=True)

    def write_conf(positions, rank):
        write_flexible_pdbqt(
            atoms, bonds, aromaticity, [],  # rigid conformer (TORSDOF 0)
            charges, conf_dir / f"conf_{rank:04d}.pdbqt",
            coords_override=[(float(positions[i][0]),
                               float(positions[i][1]),
                               float(positions[i][2])) for i in range(n)]
        )

    # Conformer 1: original structure
    write_conf(coords, 1)
    seen      = [coords.copy()]
    generated = 1

    def tree_depth(idx):
        d, cur = 0, idx
        while parent[cur] >= 0:
            cur = parent[cur]; d += 1
        return d

    weights = [tree_depth(idx) + 1 for idx in rot_indices] if rot_indices else []
    total_w = sum(weights) if weights else 1

    current      = {idx: (zmat[idx][2] or 0.0) for idx in rot_indices}
    attempts     = 0
    max_attempts = n_confs * 100
    n_clash      = 0
    n_diverse    = 0

    while generated < n_confs and attempts < max_attempts:
        attempts += 1
        if not rot_indices:
            break

        probs      = [w / total_w for w in weights]
        n_rotate   = random.randint(1, max(1, len(rot_indices) // 3))
        to_rotate  = random.choices(rot_indices, weights=probs, k=n_rotate)
        override   = dict(current)
        for idx in to_rotate:
            if random.random() < 0.3:
                override[idx] = random.uniform(-math.pi, math.pi)
            else:
                override[idx] = current[idx] + random.uniform(-math.pi/3, math.pi/3)

        new_pos = reconstruct(override)

        if _has_clash(new_pos, elems, excluded):
            n_clash += 1; continue

        # Size-scaled RMSD threshold
        n_heavy_atoms = sum(1 for e in elems if e != "H")
        if n_heavy_atoms > 100:   thresh = 0.15
        elif n_heavy_atoms > 50:  thresh = 0.25
        else:                      thresh = rmsd_threshold

        import numpy as np_inner
        too_similar = any(
            float(np_inner.sqrt(((new_pos - prev)**2).sum(axis=1).mean())) < thresh
            for prev in seen
        )
        if too_similar:
            n_diverse += 1; continue

        generated += 1
        seen.append(new_pos.copy())
        current = dict(override)
        write_conf(new_pos, generated)

        if generated % 10 == 0 or generated == n_confs:
            print(f"        Conformer {generated}/{n_confs} "
                  f"(attempts: {attempts}, clash: {n_clash}, "
                  f"diverse: {n_diverse})", flush=True)

    if generated <= 1 and n_confs > 1:
        return 1, "FALLBACK: conformational space inaccessible from input pose"

    return generated, ""


def prune_conformers(conf_dir: Path, n_heavy: int) -> int:
    """
    Prune conformers that are too similar after generation.
    Returns number of conformers kept.
    """
    import numpy as np

    if n_heavy > 100:   thresh = 0.15
    elif n_heavy > 50:  thresh = 0.25
    else:               thresh = 0.5

    conf_files = sorted(conf_dir.glob("conf_*.pdbqt"))
    if len(conf_files) <= 1:
        return len(conf_files)

    def parse_heavy_coords(pdbqt_path):
        coords = []
        with open(pdbqt_path) as f:
            for line in f:
                if not line.startswith("ATOM"): continue
                try:
                    typ = line[77:79].strip()
                    if typ in ("H", "HD"): continue
                    coords.append([float(line[30:38]),
                                   float(line[38:46]),
                                   float(line[46:54])])
                except (ValueError, IndexError):
                    continue
        return np.array(coords) if coords else None

    kept        = [conf_files[0]]
    kept_coords = [parse_heavy_coords(conf_files[0])]

    for cf in conf_files[1:]:
        c = parse_heavy_coords(cf)
        if c is None: continue
        if kept_coords[0] is not None and len(c) != len(kept_coords[0]):
            continue
        if not any(
            kc is not None and
            float(np.sqrt(((c - kc)**2).sum(axis=1).mean())) < thresh
            for kc in kept_coords
        ):
            kept.append(cf)
            kept_coords.append(c)

    keep_names = {cf.name for cf in kept}
    for cf in conf_dir.glob("conf_*.pdbqt"):
        if cf.name not in keep_names:
            cf.unlink()

    remaining = sorted(conf_dir.glob("conf_*.pdbqt"))
    for rank, cf in enumerate(remaining, 1):
        new_name = conf_dir / f"conf_{rank:04d}.pdbqt"
        if cf.name != new_name.name:
            cf.rename(new_name)

    return len(remaining)


# ══════════════════════════════════════════════════════════════════════════════
# Main processing function
# ══════════════════════════════════════════════════════════════════════════════

def process_ligand(pdb_path: Path, library_dir: Path,
                   state: LibraryState, n_confs_override: int | None = None,
                   verbose: bool = True) -> bool:
    """
    Process a single PDB ligand into the library.
    Returns True on success.
    """
    name = resolve_name(pdb_path, library_dir)

    # Claim processing slot
    if not state.claim(name):
        if verbose:
            print(f"  [SKIP] {name}: already processing or ready")
        return False

    comp_dir = library_dir / name
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Symlink original PDB
    link = comp_dir / f"{name}.pdb"
    if not link.exists():
        try:
            link.symlink_to(pdb_path.resolve())
        except Exception:
            import shutil
            shutil.copy2(pdb_path, link)

    try:
        # Parse
        atoms = parse_atoms(pdb_path)
        if not atoms:
            raise ValueError("No ATOM/HETATM records found")

        n_heavy = sum(1 for a in atoms if a["element"] != "H")
        if n_heavy < 3:
            raise ValueError(f"Too few heavy atoms ({n_heavy})")

        bonds       = build_bonds(atoms)
        aromaticity = get_aromaticity(atoms, bonds)
        rot_bonds   = get_rotatable_bonds(atoms, bonds, aromaticity)
        charges     = get_charges(pdb_path)
        props       = compute_properties(pdb_path, atoms)

        if verbose:
            print(f"  [NEW] {name}")
            print(f"        {props['formula']}  "
                  f"{props['n_heavy_atoms']} heavy  "
                  f"{len(rot_bonds)} rot  "
                  f"box≈{props['bbox']['dx']:.1f}×"
                  f"{props['bbox']['dy']:.1f}×"
                  f"{props['bbox']['dz']:.1f}Å  "
                  f"smiles={props['smiles_source']}")

        # Write flexible PDBQT
        pdbqt_path = comp_dir / f"{name}.pdbqt"
        ok = write_flexible_pdbqt(atoms, bonds, aromaticity,
                                   rot_bonds, charges, pdbqt_path)
        if not ok:
            raise ValueError("PDBQT generation failed")

        # Build and save hydrogen template for post-docking reconstruction
        try:
            from reconstruct_hydrogens import save_ligand_h_template
            save_ligand_h_template(pdb_path, comp_dir / "hydrogen_template.json")
        except Exception:
            pass  # Non-fatal — complex will just have no ligand H reconstruction

        # Conformer generation
        n_target = (n_confs_override if n_confs_override is not None
                    else default_conformer_count(n_heavy))

        conf_dir        = comp_dir / "conformers"
        n_confs_gen     = 0
        use_flex_fallback = False

        if n_target > 0:
            n_confs_gen, conf_err = generate_conformers(
                pdb_path, atoms, bonds, aromaticity,
                rot_bonds, charges, conf_dir, n_target
            )
            if conf_err.startswith("FALLBACK"):
                if verbose:
                    print(f"        [INFO] Conformers: inaccessible — "
                          f"using flexible PDBQT docking")
                use_flex_fallback = True
                n_confs_gen       = 0
            elif conf_err:
                if verbose:
                    print(f"        [WARN] Conformers: {conf_err[:80]}")
            else:
                # Prune duplicates
                n_before    = n_confs_gen
                n_confs_gen = prune_conformers(conf_dir, n_heavy)
                n_pruned    = n_before - n_confs_gen
                if n_confs_gen <= 1:
                    use_flex_fallback = True
                    n_confs_gen       = 0
                    if verbose:
                        print(f"        [INFO] All conformers pruned — "
                              f"using flexible PDBQT docking")
                elif verbose:
                    msg = (f"        Conformers: {n_confs_gen} kept "
                           f"({n_pruned} pruned)" if n_pruned > 0
                           else f"        Conformers: {n_confs_gen} generated")
                    print(msg)

        # Write state
        state.mark_ready(name, {
            "pdbqt_path":       str(pdbqt_path),
            "n_conformers":     n_confs_gen,
            "conf_dir":         str(conf_dir) if n_confs_gen > 0 else None,
            "use_flex_fallback": int(use_flex_fallback),
            "n_heavy_atoms":    props["n_heavy_atoms"],
            "n_hydrogens":      props["n_hydrogens"],
            "n_rot_bonds":      len(rot_bonds),
            "formula":          props["formula"],
            "smiles":           props["smiles"],
            "smiles_source":    props["smiles_source"],
            "bbox_json":        json.dumps(props["bbox"]),
            "elements_json":    json.dumps(props["elements"]),
        })

        # Write legacy library.json for backward compatibility
        state.write_json(library_dir / "library.json")
        return True

    except Exception as e:
        import traceback
        err = f"{e}\n{traceback.format_exc()[:300]}"
        state.mark_failed(name, str(e))
        if verbose:
            print(f"  [FAIL] {name}: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Process PDB ligands into the library.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("--job",         help="Job JSON file (all params inside)")
    p.add_argument("--ligand_dir",  default=None,
                   help="Directory containing input PDB files")
    p.add_argument("--library_dir", default=None,
                   help="Library output directory (contains library_state.db)")
    p.add_argument("--pattern",     default="*.pdb",
                   help="Glob pattern for PDB files (default: *.pdb)")
    p.add_argument("--conformers",  type=int, default=None,
                   help="Override conformer count (0 = disable)")
    p.add_argument("--refresh",     nargs="*", metavar="NAME",
                   help="Force reprocess specific ligands (no args = all)")
    p.add_argument("--refresh_all", action="store_true",
                   help="Force reprocess all ligands")
    p.add_argument("--quiet",       action="store_true")
    args = p.parse_args()

    # Overlay job file values
    if args.job:
        with open(args.job) as jf:
            job_data = json.load(jf)
        opts = job_data.get("options", {})
        if job_data.get("ligand_dir"):  args.ligand_dir  = job_data["ligand_dir"]
        if job_data.get("library_dir"): args.library_dir = job_data["library_dir"]
        if opts.get("refresh_all"):     args.refresh_all = True
        c = opts.get("conformers")
        if c is not None and c != "None" and c != "":
            args.conformers = int(c)

    if not args.ligand_dir or not args.library_dir:
        print("ERROR: --ligand_dir and --library_dir required (or --job)", file=sys.stderr)
        sys.exit(1)

    ligand_dir  = Path(args.ligand_dir)
    library_dir = Path(args.library_dir)
    verbose     = not args.quiet

    if not ligand_dir.exists():
        print(f"ERROR: ligand_dir not found: {ligand_dir}", file=sys.stderr)
        sys.exit(1)

    library_dir.mkdir(parents=True, exist_ok=True)
    state = LibraryState(library_dir)

    pdb_files = sorted(ligand_dir.glob(args.pattern))
    if not pdb_files:
        print(f"No PDB files found in {ligand_dir}")
        sys.exit(0)

    # Determine which names to force-refresh
    force_all   = args.refresh_all or (args.refresh is not None and len(args.refresh) == 0)
    force_names = set(args.refresh) if args.refresh else set()

    if verbose:
        print(f"\n=== LIGAND LIBRARY PREP ===")
        print(f"  Source:  {ligand_dir}  ({len(pdb_files)} PDB files)")
        print(f"  Library: {library_dir}")
        mode = "Force ALL" if force_all else (
               f"Force {force_names}" if force_names else "Incremental")
        print(f"  Mode:    {mode}\n")

    n_new = n_skip = n_fail = 0

    for pdb_path in pdb_files:
        name = resolve_name(pdb_path, library_dir)
        h    = file_hash(pdb_path)

        # Force refresh — reset to pending so it gets reprocessed
        if force_all or name in force_names:
            import sqlite3 as _sq
            con = _sq.connect(str(state.db_path), timeout=10)
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("UPDATE ligands SET status='pending' WHERE name=?", (name,))
            con.commit()
            con.close()

        state.register(name, h)
        rec = state.get(name)

        if rec and rec["status"] == "ready" and not force_all and name not in force_names:
            if verbose:
                print(f"  [SKP] {name}")
            n_skip += 1
            continue

        ok = process_ligand(pdb_path, library_dir, state,
                            n_confs_override=args.conformers,
                            verbose=verbose)
        if ok:
            n_new += 1
        else:
            n_fail += 1

    if verbose:
        print(f"\n=== SUMMARY ===")
        print(f"  Processed: {n_new}")
        print(f"  Skipped:   {n_skip}")
        print(f"  Failed:    {n_fail}")
        print(f"  Library:   {len(state.get_ready())} ready")
        print(f"  State:     {state.db_path}")


if __name__ == "__main__":
    random.seed(42)
    main()
