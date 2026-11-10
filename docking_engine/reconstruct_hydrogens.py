#!/usr/bin/env python3
"""
reconstruct_hydrogens.py
------------------------
Reconstruct hydrogen positions analytically after docking.

AutoDock-GPU moves heavy atoms during the search but propagates hydrogen
positions rigidly from the input PDBQT's internal coordinates, so hydrogens
in the DLG output frequently have unphysical geometry relative to their
(moved) parent heavy atom. Those hydrogens are stripped and rebuilt here
from the docked heavy-atom positions.

Geometry convention
-------------------
Every placement routine receives absolute coordinates and works with bond
directions pointing FROM the central atom TOWARD each substituent:

    d_i = normalise(neighbour_i - center)

A hydrogen direction h satisfies  dot(h, d_i) = cos(theta)  where theta is
the ideal H-X-neighbour angle (109.47 sp3, 120 sp2, 180 sp). Mixing this
convention with its inverse is what previously placed aromatic hydrogens
inside their own ring and methyl hydrogens at the tetrahedral supplement.
"""

import json
import math
import numpy as np
from pathlib import Path


# ── Standard bond lengths (Angstrom) ─────────────────────────────────────────
_BL = {"C-H": 1.090, "N-H": 1.010, "O-H": 0.960,
       "S-H": 1.340, "P-H": 1.420}

def _bl(elem_parent: str) -> float:
    return _BL.get(f"{elem_parent.upper()}-H", 1.090)


# ── Ideal angles ─────────────────────────────────────────────────────────────
_TET = math.radians(109.4712)   # sp3 X-C-X
_TRI = math.radians(120.0)      # sp2 X-C-X


# ── Vector helpers ───────────────────────────────────────────────────────────

def _norm(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def _perp(v):
    """Any unit vector perpendicular to v."""
    v = _norm(v)
    ref = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return _norm(np.cross(v, ref))


def _rotate(v, axis, angle):
    """Rodrigues rotation of v about axis by angle (radians)."""
    axis = _norm(axis)
    return (v * math.cos(angle)
            + np.cross(axis, v) * math.sin(angle)
            + axis * np.dot(axis, v) * (1.0 - math.cos(angle)))


def _cone(axis_away, perp_ref, theta_from_axis, count, start_phi=0.0):
    """
    Unit vectors on a cone of half-angle theta_from_axis about axis_away,
    evenly spaced in azimuth. Used for freely-rotating groups (methyl) and
    for the general multi-hydrogen case.
    """
    axis_away = _norm(axis_away)
    perp_ref  = _norm(perp_ref - np.dot(perp_ref, axis_away) * axis_away)
    out = []
    for i in range(count):
        phi = start_phi + 2.0 * math.pi * i / count
        p   = _rotate(perp_ref, axis_away, phi)
        out.append(_norm(axis_away * math.cos(theta_from_axis)
                         + p * math.sin(theta_from_axis)))
    return out


# ── sp3 (tetrahedral) ────────────────────────────────────────────────────────

def _place_tetrahedral(center, neighbors, bond_len, n_h):
    """
    neighbors: absolute positions of bonded heavy atoms.
    Directions d_i point from center toward each neighbour.
    """
    d = [_norm(np.asarray(nb, dtype=float) - center) for nb in neighbors]

    if len(d) >= 3 and n_h == 1:
        # Fourth vertex: opposite the sum of the three existing bonds.
        return [center + _norm(-(d[0] + d[1] + d[2])) * bond_len]

    if len(d) == 2 and n_h == 2:
        # The two H lie in the plane bisecting, and perpendicular to, the
        # X-C-Y plane, at half the tetrahedral angle from the bisector.
        bisect = _norm(-(d[0] + d[1]))
        normal = np.cross(d[0], d[1])
        normal = _perp(d[0]) if np.linalg.norm(normal) < 1e-9 else _norm(normal)
        half   = _TET / 2.0
        return [center + _norm(bisect * math.cos(half) + normal * math.sin(half)) * bond_len,
                center + _norm(bisect * math.cos(half) - normal * math.sin(half)) * bond_len]

    if len(d) == 1 and n_h >= 1:
        # Cone about the direction AWAY from the single substituent. The
        # H-C-X angle is _TET, so the angle from the away-axis is pi - _TET.
        axis  = -d[0]
        theta = math.pi - _TET
        dirs  = _cone(axis, _perp(axis), theta, max(n_h, 3))[:n_h]
        return [center + v * bond_len for v in dirs]

    if len(d) == 0:
        base = [np.array([1.,1.,1.]), np.array([-1.,-1.,1.]),
                np.array([-1.,1.,-1.]), np.array([1.,-1.,-1.])]
        return [center + _norm(b) * bond_len for b in base[:n_h]]

    # Fallback: spread remaining vertices away from the mean bond direction.
    axis = -_norm(sum(d))
    return [center + v * bond_len
            for v in _cone(axis, _perp(axis), math.pi - _TET, n_h)]


# ── sp2 (trigonal planar) ────────────────────────────────────────────────────

def _place_trigonal(center, neighbors, bond_len, n_h):
    d = [_norm(np.asarray(nb, dtype=float) - center) for nb in neighbors]

    if len(d) >= 2 and n_h == 1:
        # Third vertex: in-plane, opposite the sum of the two existing bonds.
        return [center + _norm(-(d[0] + d[1])) * bond_len]

    if len(d) == 1 and n_h == 2:
        # Two H at 120 deg from the single substituent, coplanar with it.
        normal = _perp(d[0])
        return [center + _norm(d[0] * math.cos(_TRI) + p * math.sin(_TRI)) * bond_len
                for p in (normal, -normal)]

    if len(d) == 1 and n_h == 1:
        # Angle is fixed at 120 deg; the rotation about d is unconstrained
        # without a second reference atom (see C_prev handling for amides).
        normal = _perp(d[0])
        return [center + _norm(d[0] * math.cos(_TRI) + normal * math.sin(_TRI)) * bond_len]

    if len(d) == 0:
        return [center + _norm(v) * bond_len
                for v in _cone(np.array([0.,0.,1.]), np.array([1.,0.,0.]),
                               math.pi / 2, n_h)]

    axis = -_norm(sum(d))
    return [center + v * bond_len
            for v in _cone(axis, _perp(axis), math.pi - _TRI, n_h)]


# ── sp (linear) ──────────────────────────────────────────────────────────────

def _place_linear(center, neighbors, bond_len, n_h):
    if neighbors:
        d = _norm(np.asarray(neighbors[0], dtype=float) - center)
        return [center - d * bond_len]
    return [center + np.array([bond_len, 0.0, 0.0])]


def _place_hydrogens(center, heavy_neighbors, hybridization, bond_len, n_h):
    """Dispatch on hybridisation. center and neighbours are absolute coords."""
    if n_h <= 0:
        return []
    center = np.asarray(center, dtype=float)
    if hybridization == "sp2":
        return _place_trigonal(center, heavy_neighbors, bond_len, n_h)
    if hybridization == "sp":
        return _place_linear(center, heavy_neighbors, bond_len, n_h)
    return _place_tetrahedral(center, heavy_neighbors, bond_len, n_h)


# ══════════════════════════════════════════════════════════════════════════════
# Ligand hydrogen template
# ══════════════════════════════════════════════════════════════════════════════

def build_ligand_h_template(pdb_path: Path) -> dict:
    """
    Record, from the ORIGINAL ligand PDB, which hydrogens hang off which heavy
    atom, their measured bond lengths, the parent's heavy-atom neighbours, and
    the implied hybridisation. Geometry is rebuilt from the docked heavy atoms;
    only the connectivity and bond lengths come from here.
    """
    atoms = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                aname = line[12:16].strip()
                elem  = line[76:78].strip() if len(line) > 76 else ""
                if not elem:
                    raw  = "".join(c for c in aname if c.isalpha())
                    elem = raw[0].upper() if raw else "C"
                xyz = np.array([float(line[30:38]), float(line[38:46]),
                                float(line[46:54])])
            except (ValueError, IndexError):
                continue
            atoms[aname] = {"pos": xyz, "element": elem.upper(), "aname": aname}

    if not atoms:
        return {}

    names  = list(atoms)
    coords = np.array([atoms[n]["pos"] for n in names])
    bonds  = {n: [] for n in names}
    for i, ni in enumerate(names):
        for j in range(i + 1, len(names)):
            nj = names[j]
            ei, ej = atoms[ni]["element"], atoms[nj]["element"]
            cutoff = 1.35 if ("H" in (ei, ej)) else 2.0
            if float(np.linalg.norm(coords[i] - coords[j])) < cutoff:
                bonds[ni].append(nj)
                bonds[nj].append(ni)

    template = {}
    for aname, atom in atoms.items():
        if atom["element"] == "H":
            continue
        hs = [n for n in bonds[aname] if atoms[n]["element"] == "H"]
        if not hs:
            continue
        heavy = [n for n in bonds[aname] if atoms[n]["element"] != "H"]
        total = len(heavy) + len(hs)
        hyb   = "sp" if total <= 2 else ("sp2" if total == 3 else "sp3")
        template[aname] = {
            "element":         atom["element"],
            "hybridization":   hyb,
            "hydrogens":       [{"aname": h,
                                 "bond_len": round(float(np.linalg.norm(
                                     atom["pos"] - atoms[h]["pos"])), 4)}
                                for h in hs],
            "heavy_neighbors": heavy,
        }
    return template


def save_ligand_h_template(pdb_path: Path, template_path: Path):
    template = build_ligand_h_template(pdb_path)
    template_path.write_text(json.dumps(template, indent=2))
    return template


# ══════════════════════════════════════════════════════════════════════════════
# Standard amino acid hydrogen geometry
# ══════════════════════════════════════════════════════════════════════════════

# Per-residue: {aname_heavy: (hybridization, [h_aname, ...], heavy_neighbor_names)}
# H names follow AMBER/PDB convention.
_AA_H_TEMPLATE = {
    # Backbone (common to all residues)
    "__backbone__": {
        "N":  ("sp3", ["H"],         ["CA", "C_prev"]),  # amide N
        "CA": ("sp3", ["HA"],        ["N", "C", "CB"]),
        "C":  ("sp2", [],            ["CA", "O", "N_next"]),  # no H on carbonyl C
    },
    "ALA": {
        "CB": ("sp3", ["HB1","HB2","HB3"], ["CA"]),
    },
    "ARG": {
        "CB": ("sp3", ["HB2","HB3"],       ["CA","CG"]),
        "CG": ("sp3", ["HG2","HG3"],       ["CB","CD"]),
        "CD": ("sp3", ["HD2","HD3"],       ["CG","NE"]),
        "NE": ("sp2", ["HE"],              ["CD","CZ"]),
        "NH1":("sp2", ["HH11","HH12"],     ["CZ"]),
        "NH2":("sp2", ["HH21","HH22"],     ["CZ"]),
    },
    "ASN": {
        "CB":  ("sp3", ["HB2","HB3"],   ["CA","CG"]),
        "ND2": ("sp2", ["HD21","HD22"], ["CG"]),
    },
    "ASP": {
        "CB": ("sp3", ["HB2","HB3"], ["CA","CG"]),
    },
    "CYS": {
        "CB": ("sp3", ["HB2","HB3"], ["CA","SG"]),
        "SG": ("sp3", ["HG"],        ["CB"]),
    },
    "GLN": {
        "CB":  ("sp3", ["HB2","HB3"],   ["CA","CG"]),
        "CG":  ("sp3", ["HG2","HG3"],   ["CB","CD"]),
        "NE2": ("sp2", ["HE21","HE22"], ["CD"]),
    },
    "GLU": {
        "CB": ("sp3", ["HB2","HB3"], ["CA","CG"]),
        "CG": ("sp3", ["HG2","HG3"], ["CB","CD"]),
    },
    "GLY": {
        "CA": ("sp3", ["HA2","HA3"], ["N","C"]),  # override backbone HA
    },
    "HIS": {
        "CB":  ("sp3", ["HB2","HB3"], ["CA","CG"]),
        "ND1": ("sp2", ["HD1"],       ["CG","CE1"]),
        "CD2": ("sp2", ["HD2"],       ["CG","NE2"]),
        "CE1": ("sp2", ["HE1"],       ["ND1","NE2"]),
    },
    "ILE": {
        "CB":  ("sp3", ["HB"],           ["CA","CG1","CG2"]),
        "CG1": ("sp3", ["HG12","HG13"],  ["CB","CD1"]),
        "CG2": ("sp3", ["HG21","HG22","HG23"], ["CB"]),
        "CD1": ("sp3", ["HD11","HD12","HD13"], ["CG1"]),
    },
    "LEU": {
        "CB":  ("sp3", ["HB2","HB3"],          ["CA","CG"]),
        "CG":  ("sp3", ["HG"],                 ["CB","CD1","CD2"]),
        "CD1": ("sp3", ["HD11","HD12","HD13"], ["CG"]),
        "CD2": ("sp3", ["HD21","HD22","HD23"], ["CG"]),
    },
    "LYS": {
        "CB": ("sp3", ["HB2","HB3"],           ["CA","CG"]),
        "CG": ("sp3", ["HG2","HG3"],           ["CB","CD"]),
        "CD": ("sp3", ["HD2","HD3"],           ["CG","CE"]),
        "CE": ("sp3", ["HE2","HE3"],           ["CD","NZ"]),
        "NZ": ("sp3", ["HZ1","HZ2","HZ3"],     ["CE"]),
    },
    "MET": {
        "CB": ("sp3", ["HB2","HB3"],           ["CA","CG"]),
        "CG": ("sp3", ["HG2","HG3"],           ["CB","SD"]),
        "CE": ("sp3", ["HE1","HE2","HE3"],     ["SD"]),
    },
    "PHE": {
        "CB":  ("sp3", ["HB2","HB3"], ["CA","CG"]),
        "CD1": ("sp2", ["HD1"],       ["CG","CE1"]),
        "CD2": ("sp2", ["HD2"],       ["CG","CE2"]),
        "CE1": ("sp2", ["HE1"],       ["CD1","CZ"]),
        "CE2": ("sp2", ["HE2"],       ["CD2","CZ"]),
        "CZ":  ("sp2", ["HZ"],        ["CE1","CE2"]),
    },
    "PRO": {
        "CB": ("sp3", ["HB2","HB3"], ["CA","CG"]),
        "CG": ("sp3", ["HG2","HG3"], ["CB","CD"]),
        "CD": ("sp3", ["HD2","HD3"], ["CG","N"]),
        # PRO N has no H
    },
    "SER": {
        "CB": ("sp3", ["HB2","HB3"], ["CA","OG"]),
        "OG": ("sp3", ["HG"],        ["CB"]),
    },
    "THR": {
        "CB":  ("sp3", ["HB"],                  ["CA","OG1","CG2"]),
        "OG1": ("sp3", ["HG1"],                 ["CB"]),
        "CG2": ("sp3", ["HG21","HG22","HG23"],  ["CB"]),
    },
    "TRP": {
        "CB":  ("sp3", ["HB2","HB3"], ["CA","CG"]),
        "CD1": ("sp2", ["HD1"],       ["CG","NE1"]),
        "NE1": ("sp2", ["HE1"],       ["CD1","CE2"]),
        "CE3": ("sp2", ["HE3"],       ["CD2","CZ3"]),
        "CZ2": ("sp2", ["HZ2"],       ["CE2","CH2"]),
        "CZ3": ("sp2", ["HZ3"],       ["CE3","CH2"]),
        "CH2": ("sp2", ["HH2"],       ["CZ2","CZ3"]),
    },
    "TYR": {
        "CB":  ("sp3", ["HB2","HB3"], ["CA","CG"]),
        "CD1": ("sp2", ["HD1"],       ["CG","CE1"]),
        "CD2": ("sp2", ["HD2"],       ["CG","CE2"]),
        "CE1": ("sp2", ["HE1"],       ["CD1","CZ"]),
        "CE2": ("sp2", ["HE2"],       ["CD2","CZ"]),
        "OH":  ("sp3", ["HH"],        ["CZ"]),
    },
    "VAL": {
        "CB":  ("sp3", ["HB"],                  ["CA","CG1","CG2"]),
        "CG1": ("sp3", ["HG11","HG12","HG13"],  ["CB"]),
        "CG2": ("sp3", ["HG21","HG22","HG23"],  ["CB"]),
    },
}

# Aliases for common variant names
_AA_ALIASES = {
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    "HIE": "HIS", "HID": "HIS", "HIP": "HIS",
    "CYX": "CYS", "CYM": "CYS",
    "GLH": "GLU", "ASH": "ASP",
    "LYN": "LYS", "ARN": "ARG",
    "MSE": "MET",
}



# Backbone amide N is sp2, not sp3 — it is conjugated with the preceding
# carbonyl. Placing it as sp3 without a second reference atom was what
# produced ~47 deg CA-N-H angles.
_AA_H_TEMPLATE["__backbone__"]["N"]  = ("sp2", ["H"],  ["CA", "C_prev"])
_AA_H_TEMPLATE["__backbone__"]["CA"] = ("sp3", ["HA"], ["N", "C", "CB"])


# ══════════════════════════════════════════════════════════════════════════════
# PDB record formatting
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_atom(record, serial, aname, resname, chain, resseq, xyz, element="H"):
    """
    Emit a PDB ATOM/HETATM line with correct column placement.
    Atom names of three characters or fewer are indented one column so the
    element symbol occupies columns 13-14, per PDB convention.
    """
    name = aname if len(aname) >= 4 else f" {aname:<3s}"
    return (f"{record:<6s}{serial:5d} {name:<4s} "
            f"{resname:>3s} {chain:1s}{resseq:>4s}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
            f"  1.00  0.00          {element:>2s}\n")


def _element_of(aname: str) -> str:
    a = aname.strip()
    if a[:1].isdigit():
        a = a[1:]
    if a[:2].upper() in ("CL", "BR", "MG", "ZN", "FE", "MN", "NA", "CA"):
        return a[:2].capitalize()
    return a[:1].upper() if a else "C"


# ══════════════════════════════════════════════════════════════════════════════
# Residue hydrogen reconstruction
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_residue_hydrogens(residue_lines, resname, prev_c=None):
    """
    Build hydrogen positions for one residue from its heavy atoms.

    residue_lines: ATOM lines for this residue, heavy atoms only.
    prev_c:        position of the preceding residue's carbonyl carbon, used
                   to define the peptide plane for the backbone amide H.
                   Without it the amide H angle is still correct but its
                   rotation about N-CA is arbitrary.

    Returns a list of (aname, position) pairs.
    """
    canonical = _AA_ALIASES.get(resname, resname)
    res_tmpl  = _AA_H_TEMPLATE.get(canonical, {})
    backbone  = _AA_H_TEMPLATE.get("__backbone__", {})

    pos = {}
    for line in residue_lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            pos[line[12:16].strip()] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except (ValueError, IndexError):
            continue
    if not pos:
        return []

    placed = []

    def run(aname, entry):
        hyb, h_names, nb_names = entry
        if aname not in pos or not h_names:
            return
        neighbors = []
        for n in nb_names:
            if n == "C_prev":
                if prev_c is not None:
                    neighbors.append(prev_c)
            elif n == "N_next":
                continue
            elif n in pos:
                neighbors.append(pos[n])
        bond = _bl(_element_of(aname))
        for h_name, p in zip(h_names,
                             _place_hydrogens(pos[aname], neighbors,
                                              hyb, bond, len(h_names))):
            placed.append((h_name, p))

    # Backbone: CA first (GLY has two HA), then the amide NH (absent on PRO).
    if canonical == "GLY" and "CA" in res_tmpl:
        run("CA", res_tmpl["CA"])
    elif "CA" in backbone:
        run("CA", backbone["CA"])

    if canonical != "PRO" and "N" in backbone:
        run("N", backbone["N"])

    # Side chain
    for aname, entry in res_tmpl.items():
        if canonical == "GLY" and aname == "CA":
            continue
        run(aname, entry)

    return placed


def reconstruct_ligand_hydrogens(heavy_lines, template):
    """Returns a list of (aname, position) pairs for the ligand."""
    pos = {}
    for line in heavy_lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            pos[line[12:16].strip()] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except (ValueError, IndexError):
            continue

    placed = []
    for heavy, entry in (template or {}).items():
        if heavy not in pos:
            continue
        h_defs = entry.get("hydrogens") or []
        if not h_defs:
            continue
        neighbors = [pos[n] for n in entry.get("heavy_neighbors", []) if n in pos]
        bond = sum(h["bond_len"] for h in h_defs) / len(h_defs)
        for h_def, p in zip(h_defs,
                            _place_hydrogens(pos[heavy], neighbors,
                                             entry.get("hybridization", "sp3"),
                                             bond, len(h_defs))):
            placed.append((h_def["aname"], p))
    return placed


# ══════════════════════════════════════════════════════════════════════════════
# Whole-complex reconstruction
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_complex_hydrogens(complex_pdb: Path,
                                   ligand_template: dict | None = None) -> None:
    """
    Rewrite a hydrogen-stripped complex PDB with hydrogens rebuilt in place.

    Hydrogens are emitted immediately after their parent residue's heavy
    atoms and serials are renumbered sequentially. Appending them as a
    trailing block, as a previous version did, leaves every hydrogen
    non-contiguous with its residue, which stops viewers from bonding them.
    """
    rec_lines, lig_lines = [], []
    for line in Path(complex_pdb).read_text().splitlines(keepends=True):
        if line.startswith("ATOM"):
            rec_lines.append(line)
        elif line.startswith("HETATM"):
            lig_lines.append(line)

    # Group receptor atoms by residue, preserving file order.
    order, groups = [], {}
    for line in rec_lines:
        key = (line[21], line[22:26].strip(), line[17:20].strip())
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(line)

    # Carbonyl C of each residue, for the next residue's amide H.
    def carbonyl_c(lines):
        for l in lines:
            if l[12:16].strip() == "C":
                try:
                    return np.array([float(l[30:38]), float(l[38:46]),
                                     float(l[46:54])])
                except (ValueError, IndexError):
                    return None
        return None

    out, serial = [], 0
    prev_key, prev_c = None, None

    for key in order:
        chain, resseq, resname = key
        lines = groups[key]

        # Only treat as a peptide neighbour if same chain and consecutive.
        use_prev = None
        if prev_key is not None and prev_key[0] == chain:
            try:
                if int(resseq) - int(prev_key[1]) == 1:
                    use_prev = prev_c
            except ValueError:
                pass

        for l in lines:
            serial += 1
            out.append(l[:6] + f"{serial:5d}" + l[11:])

        for aname, p in reconstruct_residue_hydrogens(lines, resname, use_prev):
            serial += 1
            out.append(_fmt_atom("ATOM", serial, aname, resname,
                                 chain, resseq, p, "H"))

        prev_key, prev_c = key, carbonyl_c(lines)

    out.append("TER\n")

    # Ligand: heavy atoms then its reconstructed hydrogens.
    for l in lig_lines:
        serial += 1
        out.append(l[:6] + f"{serial:5d}" + l[11:])

    if ligand_template and lig_lines:
        lig_res   = lig_lines[0][17:20].strip() or "LIG"
        lig_chain = lig_lines[0][21]
        lig_seq   = lig_lines[0][22:26].strip()
        for aname, p in reconstruct_ligand_hydrogens(lig_lines, ligand_template):
            serial += 1
            out.append(_fmt_atom("HETATM", serial, aname, lig_res,
                                 lig_chain, lig_seq, p, "H"))

    out.append("END\n")
    Path(complex_pdb).write_text("".join(out))
