#!/usr/bin/env python3
"""
complex_validity.py - is a docked complex physically/chemically reasonable?

A good AutoDock score does not mean a usable pose. The two failure modes this
tool catches are the ones a score cannot see:

  clash         Heavy atoms of the ligand overlapping heavy atoms of the
                receptor, measured as a fraction of the van der Waals sum
                (the 0.75 pipeline clash metric). A count of clashing pairs.

  ring-piercing A bond threaded THROUGH a ring - a ligand bond passing inside
                the disk of a receptor aromatic ring (Phe/Tyr/Trp/His) or of
                another ligand ring, or a ligand bond through one of the
                ligand's own rings. This is topologically impossible in a real
                structure, scores can look fine, and a pairwise-distance clash
                test often misses it because no two atoms quite overlap.

A pose that clashes or pierces a ring is rejected regardless of score.

Usage:
    python3 complex_validity.py --complex COMPLEX.pdb [options]
    python3 complex_validity.py --receptor REC.pdb --ligand LIG.pdb [options]

Options:
    --complex PATH        Single PDB holding receptor (ATOM) and ligand
    --receptor PATH       Receptor PDB (use with --ligand)
    --ligand PATH         Ligand PDB (use with --receptor, or HETATM in complex)
    --ligand-resname RES  Residue name marking ligand records in --complex
                          (default: LIG)
    --clash-scale FRAC    Fraction of vdW sum below which a pair clashes
                          (default: 0.75, the pipeline clash metric)
    --clash-max N         Allow up to N clashing pairs before failing
                          (default: 0)
    --ring-margin AA      Extra radius added to each ring when testing a
                          piercing, in angstrom (default: 0.0)
    --max-ring N          Largest ligand ring size to perceive (default: 8)
    --json PATH           Write the full verdict as JSON
    --quiet               Print only the one-line verdict
    --help                Show this message

Exit status is 0 for a valid complex, 1 for an invalid one, 2 on error - so it
can gate a pipeline step directly.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# ── van der Waals radii, keyed by AD4 atom type (the 0.75 clash metric) ───────
_VDW = {"A": 1.70, "C": 1.70, "N": 1.55, "NA": 1.55, "O": 1.52, "OA": 1.52,
        "S": 1.80, "SA": 1.80, "HD": 1.20, "H": 1.20, "P": 1.80,
        "F": 1.47, "Cl": 1.75, "Br": 1.85, "I": 1.98,
        "Zn": 1.39, "Fe": 1.32, "Mg": 1.73, "Ca": 1.74, "Mn": 1.61}
_CLASH_SCALE = 0.75

# ── covalent radii (angstrom), keyed by element, for bond inference ───────────
_COV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07,
        "F": 0.57, "CL": 1.02, "BR": 1.20, "I": 1.39,
        "ZN": 1.22, "FE": 1.32, "MG": 1.41, "CA": 1.76, "MN": 1.39}
_BOND_TOL = 0.45     # dist <= rcov_i + rcov_j + tol  => bonded

# ── receptor ring templates: residue -> list of rings (each a tuple of atom
#    names). Aromatic side chains plus the proline ring. Covers the common
#    protonation-state aliases for histidine. ─────────────────────────────────
_RING_TEMPLATES = {
    "PHE": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")],
    "TYR": [("CG", "CD1", "CD2", "CE1", "CE2", "CZ")],
    "HIS": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "HID": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "HIE": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "HIP": [("CG", "ND1", "CD2", "CE1", "NE2")],
    "TRP": [("CG", "CD1", "CD2", "NE1", "CE2"),
            ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2")],
    "PRO": [("N", "CA", "CB", "CG", "CD")],
}


def _ad4_to_element(ad4_type: str, atom_name: str = "") -> str:
    """
    Map an AD4 atom type (or, failing that, a PDB atom name) to a plain
    element symbol in upper case.

    Input:  ad4_type  - string from PDB columns 77-78 (e.g. "OA", "A", "HD")
            atom_name  - PDB columns 13-16, used only if ad4_type is blank
    Output: element symbol such as "C", "N", "O", "CL"
    Behaviour: AD4 aromatic carbon "A" and oxygen "OA"/sulfur "SA"/nitrogen
               "NA" collapse to their base element; otherwise the leading
               alphabetic run of whichever string is available is returned.
    """
    t = (ad4_type or "").strip()
    if t:
        if t in ("A",):
            return "C"
        if t in ("OA",):
            return "O"
        if t in ("NA",):
            return "N"
        if t in ("SA",):
            return "S"
        if t in ("HD",):
            return "H"
        if len(t) == 2 and t.upper() in ("CL", "BR", "MG", "ZN", "FE", "CA", "MN"):
            return t.upper()
        return t[0].upper()
    raw = "".join(c for c in (atom_name or "") if c.isalpha())
    return raw[0].upper() if raw else "C"


# ══════════════════════════════════════════════════════════════════════════════
# PDB reading
# ══════════════════════════════════════════════════════════════════════════════

def read_complex(path: Path, ligand_resname: str = "LIG"):
    """
    Read a complex PDB into receptor and ligand atom records.

    Input:  path            - PDB file with receptor as ATOM and ligand as
                              HETATM (or ATOM) whose residue name is
                              ligand_resname
            ligand_resname  - residue name that marks ligand atoms
    Output: (receptor, ligand) where each is a list of dicts with keys
            xyz (np.array(3)), ad4 (type), elem, name, resname, resid, is_h
    Behaviour: HETATM/ATOM records matching ligand_resname become the ligand;
               all other ATOM records become the receptor. Hydrogens are kept
               but flagged, since clash counting ignores them while bond
               inference needs the heavy-atom graph only.
    """
    receptor, ligand = [], []
    for line in _atom_lines(path):
        rec = _parse_atom(line)
        if rec is None:
            continue
        if rec["resname"] == ligand_resname:
            ligand.append(rec)
        elif line.startswith("ATOM"):
            receptor.append(rec)
    return receptor, ligand


def read_single(path: Path):
    """Read every ATOM/HETATM record of a standalone PDB as one atom list."""
    return [r for r in (_parse_atom(l) for l in _atom_lines(path)) if r]


def _atom_lines(path: Path):
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                yield line


def _parse_atom(line: str):
    """Parse one PDB ATOM/HETATM line into an atom record, or None on failure."""
    try:
        name = line[12:16].strip()
        resname = line[17:20].strip()
        resid = line[22:27].strip()        # resSeq + iCode, as a key
        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
    except (ValueError, IndexError):
        return None
    ad4 = line[76:78].strip() if len(line) > 76 else ""
    elem = _ad4_to_element(ad4, name)
    return {"xyz": np.array([x, y, z]), "ad4": ad4, "elem": elem,
            "name": name, "resname": resname, "resid": resid,
            "is_h": elem in ("H", "D")}


# ══════════════════════════════════════════════════════════════════════════════
# Clash counting — the pipeline clash metric (0.75 x vdW sum, heavy atoms)
# ══════════════════════════════════════════════════════════════════════════════

def count_clashes(ligand, receptor, scale: float = _CLASH_SCALE):
    """
    Count ligand/receptor heavy-atom pairs closer than scale * (vdW sum).

    Input:  ligand, receptor - atom-record lists from read_complex
            scale            - clash fraction (0.75, the pipeline metric)
    Output: integer count of clashing heavy-atom pairs
    Behaviour: hydrogens are skipped on both sides; vdW radii come from the
               AD4 type via the shared _VDW table so the number is directly
               comparable to the pipeline's `validation` column.
    """
    lig = [a for a in ligand if not a["is_h"]]
    rec = [a for a in receptor if not a["is_h"]]
    if not lig or not rec:
        return 0
    L = np.array([a["xyz"] for a in lig])
    R = np.array([a["xyz"] for a in rec])
    rl = np.array([_VDW.get(a["ad4"], _VDW.get(a["elem"], 1.70)) for a in lig])
    rr = np.array([_VDW.get(a["ad4"], _VDW.get(a["elem"], 1.70)) for a in rec])
    n = 0
    for i in range(len(lig)):
        d = np.sqrt(((R - L[i]) ** 2).sum(axis=1))
        n += int((d < scale * (rr + rl[i])).sum())
    return n


# ══════════════════════════════════════════════════════════════════════════════
# Bond and ring perception
# ══════════════════════════════════════════════════════════════════════════════

def infer_bonds(atoms):
    """
    Infer heavy-atom connectivity from interatomic distance.

    Input:  atoms - atom-record list (hydrogens are ignored)
    Output: (bonds, heavy_index) where bonds is a list of (i, j) index pairs
            INTO the returned heavy_index list of atom records
    Behaviour: two heavy atoms are bonded when their separation is at most the
               sum of covalent radii plus a tolerance. O(n^2) over heavy atoms,
               which is fine for a single ligand or sidechain set.
    """
    heavy = [a for a in atoms if not a["is_h"]]
    X = np.array([a["xyz"] for a in heavy]) if heavy else np.zeros((0, 3))
    rc = np.array([_COV.get(a["elem"], 0.77) for a in heavy])
    bonds = []
    for i in range(len(heavy)):
        for j in range(i + 1, len(heavy)):
            d = np.linalg.norm(X[i] - X[j])
            if 0.4 < d <= rc[i] + rc[j] + _BOND_TOL:
                bonds.append((i, j))
    return bonds, heavy


def find_rings(bonds, n_atoms: int, max_ring: int = 8):
    """
    Perceive the smallest rings in a bond graph.

    Input:  bonds   - list of (i, j) index pairs
            n_atoms - number of nodes
            max_ring- largest ring size to report
    Output: list of rings, each a list of atom indices in ring order
    Behaviour: for every bond, remove it and find the shortest path between its
               two endpoints in the remaining graph; that path plus the bond is
               the smallest ring through that bond. Rings are de-duplicated by
               their atom set. This is the standard smallest-ring-per-bond
               heuristic - enough to define ring planes for the piercing test
               without an external cheminformatics dependency.
    """
    from collections import deque, defaultdict
    adj = defaultdict(set)
    for i, j in bonds:
        adj[i].add(j); adj[j].add(i)

    seen = set()
    rings = []
    for (a, b) in bonds:
        # shortest path a->b without using the direct edge
        adj[a].discard(b); adj[b].discard(a)
        prev = {a: None}
        dq = deque([a])
        while dq:
            u = dq.popleft()
            if u == b:
                break
            for w in adj[u]:
                if w not in prev:
                    prev[w] = u
                    dq.append(w)
        adj[a].add(b); adj[b].add(a)            # restore edge
        if b not in prev:
            continue
        path = []
        u = b
        while u is not None:
            path.append(u)
            u = prev[u]
        if len(path) > max_ring:
            continue
        key = frozenset(path)
        if key in seen:
            continue
        seen.add(key)
        rings.append(path)
    return rings


def ring_plane(coords):
    """
    Best-fit plane and in-plane radius for a set of ring-atom coordinates.

    Input:  coords - (k, 3) array of ring atom positions
    Output: (centroid(3), normal(3) unit, radius) where radius is the mean
            centroid-to-atom distance (the effective disk a bond must miss)
    Behaviour: the normal is the least-variance singular vector of the centred
               coordinates (SVD), i.e. the direction of smallest spread, which
               is the plane normal for a roughly planar ring.
    """
    C = coords.mean(axis=0)
    Q = coords - C
    _, _, vt = np.linalg.svd(Q)
    normal = vt[2]
    normal = normal / (np.linalg.norm(normal) or 1.0)
    radius = float(np.linalg.norm(Q, axis=1).mean())
    return C, normal, radius


def segment_pierces_ring(P, Q, centroid, normal, radius, margin: float = 0.0):
    """
    Does the segment P->Q pass through a ring's disk?

    Input:  P, Q     - segment endpoints (3,)
            centroid, normal, radius - ring plane from ring_plane()
            margin   - extra radius allowance (angstrom)
    Output: (pierces: bool, in_plane_distance: float or None)
    Behaviour: intersect the segment with the ring plane; if the crossing
               parameter lies strictly inside the segment and the crossing
               point is within (radius + margin) of the ring centroid, the bond
               threads the ring. A segment parallel to the plane never pierces.
    """
    d = Q - P
    denom = float(np.dot(normal, d))
    if abs(denom) < 1e-9:
        return False, None                      # parallel to the ring plane
    t = float(np.dot(normal, centroid - P)) / denom
    if not (0.0 < t < 1.0):
        return False, None                      # crossing is outside the bond
    hit = P + t * d
    dist = float(np.linalg.norm(hit - centroid))
    return (dist <= radius + margin), dist


def receptor_rings(receptor):
    """
    Collect aromatic/proline ring atom sets from receptor residues.

    Input:  receptor - atom-record list
    Output: list of {"label": str, "coords": (k,3) array}
    Behaviour: groups atoms by (resname, resid), and for each residue whose
               name has a ring template, gathers the template's atoms when all
               are present. Label is e.g. "TYR 145".
    """
    from collections import defaultdict
    byres = defaultdict(dict)
    for a in receptor:
        if a["is_h"]:
            continue
        byres[(a["resname"], a["resid"])][a["name"]] = a["xyz"]
    rings = []
    for (resname, resid), atoms in byres.items():
        for template in _RING_TEMPLATES.get(resname, []):
            if all(n in atoms for n in template):
                coords = np.array([atoms[n] for n in template])
                rings.append({"label": f"{resname} {resid}",
                              "coords": coords})
    return rings


# ══════════════════════════════════════════════════════════════════════════════
# Top-level validation
# ══════════════════════════════════════════════════════════════════════════════

def find_piercings(ligand, receptor, max_ring: int = 8, margin: float = 0.0):
    """
    Every ligand bond that threads a receptor ring or a ligand ring.

    Input:  ligand, receptor - atom-record lists
            max_ring         - largest ligand ring to perceive
            margin           - extra ring radius allowance
    Output: list of {"bond": (name_i, name_j), "ring": label,
                     "distance": in-plane distance}
    Behaviour: builds the ligand heavy-atom graph, perceives ligand rings,
               then tests each ligand bond against (a) every receptor aromatic
               ring and (b) every ligand ring, skipping any ring that shares an
               atom with the bond under test (a substituent bond emanating from
               a ring is not a piercing).
    """
    lig_bonds, lig_heavy = infer_bonds(ligand)
    lig_rings_idx = find_rings(lig_bonds, len(lig_heavy), max_ring)

    rings = [{"label": r["label"], "coords": r["coords"], "atoms": None}
             for r in receptor_rings(receptor)]
    for ring in lig_rings_idx:
        coords = np.array([lig_heavy[i]["xyz"] for i in ring])
        rings.append({"label": "ligand ring " + "-".join(
            lig_heavy[i]["name"] for i in ring),
            "coords": coords, "atoms": set(ring)})

    out = []
    for (i, j) in lig_bonds:
        P = lig_heavy[i]["xyz"]; Q = lig_heavy[j]["xyz"]
        for ring in rings:
            if ring["atoms"] is not None and (i in ring["atoms"]
                                              or j in ring["atoms"]):
                continue                        # bond belongs to this ligand ring
            C, nrm, rad = ring_plane(ring["coords"])
            pierces, dist = segment_pierces_ring(P, Q, C, nrm, rad, margin)
            if pierces:
                out.append({"bond": (lig_heavy[i]["name"], lig_heavy[j]["name"]),
                            "ring": ring["label"],
                            "distance": round(dist, 2)})
    return out


def _core_contains(xyz, core_mask):
    """
    Is a point inside a run-length-encoded pocket-core mask?

    Input:  xyz       - point in the docking (box) frame
            core_mask - dict from cavity_box.rasterize_core (shape, spacing,
                        origin, runs)
    Output: bool
    Behaviour: maps the point to its grid cell and checks the sorted run table.
               Kept in sync by hand with cavity_box.core_contains (same format).
    """
    nz, ny, nx = core_mask["shape"]
    s = core_mask["spacing"]
    ox, oy, oz = core_mask["origin"]
    ix = int(math.floor((xyz[0] - ox) / s))
    iy = int(math.floor((xyz[1] - oy) / s))
    iz = int(math.floor((xyz[2] - oz) / s))
    if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
        return False
    flat = (iz * ny + iy) * nx + ix
    for start, length in core_mask["runs"]:
        if start <= flat < start + length:
            return True
        if start > flat:
            break
    return False


def core_occupancy(ligand, core_mask):
    """
    Fraction of ligand heavy atoms inside the pocket core.

    Input:  ligand    - atom-record list (hydrogens ignored)
            core_mask - the stored RLE core mask, or None
    Output: float in [0,1]; 1.0 when there is no mask (no constraint)
    Behaviour: the core mask is the pocket WITHOUT the apron, so a pose draped
               on the surface scores low occupancy even if it clears clashes.
    """
    if not core_mask:
        return 1.0
    lig = [a for a in ligand if not a["is_h"]]
    if not lig:
        return 1.0
    return sum(1 for a in lig if _core_contains(a["xyz"], core_mask)) / len(lig)


def validate(ligand, receptor, clash_scale: float = _CLASH_SCALE,
             clash_max: int = 0, max_ring: int = 8, ring_margin: float = 0.0,
             core_mask=None, min_core: float = 0.3):
    """
    Full physical-validity verdict for a ligand pose against a receptor.

    Input:  ligand, receptor - atom-record lists
            clash_scale, clash_max, max_ring, ring_margin - thresholds
            core_mask - optional RLE pocket core; when given, the pose must put
                        at least min_core of its heavy atoms inside the pocket
    Output: dict with keys clashes, clash_ok, piercings, pierce_ok,
            core_fraction, core_ok, valid
    Behaviour: the pose is valid only when clashes <= clash_max AND no bond
               threads a ring AND (if a core mask is supplied) enough of the
               ligand sits in the pocket core. Ring-piercing and surface-parking
               are never traded off against score.
    """
    clashes = count_clashes(ligand, receptor, clash_scale)
    piercings = find_piercings(ligand, receptor, max_ring, ring_margin)
    clash_ok = clashes <= clash_max
    pierce_ok = len(piercings) == 0
    core_frac = core_occupancy(ligand, core_mask)
    core_ok = (core_mask is None) or (core_frac >= min_core)
    return {"clashes": clashes, "clash_ok": clash_ok,
            "piercings": piercings, "pierce_ok": pierce_ok,
            "core_fraction": round(core_frac, 3), "core_ok": core_ok,
            "valid": clash_ok and pierce_ok and core_ok}


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        prog="complex_validity.py", add_help=False,
        description="Judge whether a docked complex is physically reasonable "
                    "(no clashes, no bonds threaded through rings).")
    p.add_argument("--complex", metavar="PATH",
                   help="Single PDB with receptor (ATOM) and ligand (HETATM)")
    p.add_argument("--receptor", metavar="PATH", help="Receptor PDB")
    p.add_argument("--ligand", metavar="PATH", help="Ligand PDB")
    p.add_argument("--ligand-resname", default="LIG", metavar="RES",
                   help="Residue name marking ligand records (default: LIG)")
    p.add_argument("--clash-scale", type=float, default=_CLASH_SCALE,
                   metavar="FRAC", help="vdW-sum fraction for a clash "
                                        "(default: 0.75)")
    p.add_argument("--clash-max", type=int, default=0, metavar="N",
                   help="Allowed clashing pairs before failing (default: 0)")
    p.add_argument("--ring-margin", type=float, default=0.0, metavar="AA",
                   help="Extra ring radius when testing a pierce (default: 0.0)")
    p.add_argument("--max-ring", type=int, default=8, metavar="N",
                   help="Largest ligand ring size to perceive (default: 8)")
    p.add_argument("--json", metavar="PATH", help="Write verdict as JSON")
    p.add_argument("--quiet", action="store_true",
                   help="Print only the one-line verdict")
    p.add_argument("--help", action="help", help="Show this message")
    args = p.parse_args()

    try:
        if args.complex:
            receptor, ligand = read_complex(Path(args.complex),
                                            args.ligand_resname)
            if not ligand:
                print(f"ERROR: no residues named {args.ligand_resname} in "
                      f"{args.complex}", file=sys.stderr)
                sys.exit(2)
        elif args.receptor and args.ligand:
            receptor = read_single(Path(args.receptor))
            ligand = read_single(Path(args.ligand))
        else:
            print("ERROR: give --complex, or both --receptor and --ligand",
                  file=sys.stderr)
            sys.exit(2)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    v = validate(ligand, receptor, args.clash_scale, args.clash_max,
                 args.max_ring, args.ring_margin)

    verdict = "VALID" if v["valid"] else "INVALID"
    if not args.quiet:
        print(f"  clashes       : {v['clashes']}  "
              f"({'ok' if v['clash_ok'] else 'FAIL'}, "
              f"max {args.clash_max})")
        print(f"  ring-piercings: {len(v['piercings'])}  "
              f"({'ok' if v['pierce_ok'] else 'FAIL'})")
        for pr in v["piercings"]:
            print(f"      bond {pr['bond'][0]}-{pr['bond'][1]} threads "
                  f"{pr['ring']}  ({pr['distance']} A from centre)")
    print(f"  VERDICT: {verdict}")

    if args.json:
        Path(args.json).write_text(json.dumps(v, indent=2))

    sys.exit(0 if v["valid"] else 1)


if __name__ == "__main__":
    main()
