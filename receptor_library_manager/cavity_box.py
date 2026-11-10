#!/usr/bin/env python3
"""
cavity_box.py - size a docking box from the receptor's measured cavity.

Replaces library-bbox autoboxing (a box big enough for the largest ligand's
bounding dimensions, which can sit anywhere on the protein) with a box fitted
to the actual pocket. The pocket is measured, not guessed:

  1. a seed point locates the site (explicit XYZ, a residue-mask centroid, or
     the native ligand's centroid);
  2. a provisional single-probe AutoGrid is built around the seed purely to
     measure burial on, then discarded;
  3. box_check grows the buried cavity from the seed and, by default, an apron
     onto the adjacent surface;
  4. the receptor is reoriented only when that shrinks the box enough to be
     worth the frame change (>= a volume-gain threshold, default 2x);
  5. the final box is the extent of the grown region, and the pocket CORE
     (cavity without apron) is stored run-length-encoded so the docking engine
     can reject poses that park in the apron without entering the pocket.

The box, the 4x4 reorientation transform, and the core mask are returned for
storage in receptor_prep.json. Applying the transform (rotating the receptor,
inverse-transforming docked poses) is a separate step.

This module owns the orchestration; box_check.py owns the cavity measurement.

Usage (standalone, for inspecting a seed/box on an existing map):
    python3 cavity_box.py --map RECEPTOR.C.map --seed X,Y,Z [options]

Options:
    --map PATH          A single AutoGrid probe map to measure on
    --seed X,Y,Z        Seed point, or 'center' for the map centre
    --pad AA            Padding added around the region extent (default: 4.0)
    --apron-reach AA    Surface apron reach; 0 disables the apron (default: 8.0)
    --reorient-gain G   Reorient only if it shrinks the box by >= G (default: 2.0)
    --json PATH         Write the measured box/transform/core as JSON
    --help              Show this message
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import box_check as bc


# ══════════════════════════════════════════════════════════════════════════════
# Seed resolution
# ══════════════════════════════════════════════════════════════════════════════

_SKIP_HET = {"HOH", "WAT", "DOD", "NA", "CL", "K", "MG", "ZN", "CA", "MN",
             "SO4", "PO4", "GOL", "EDO", "ACT"}


def _read_atoms(pdb_path):
    """
    Read ATOM/HETATM records as dicts.

    Input:  pdb_path - a PDB file
    Output: list of {rec, resname, name, chain, resseq, xyz}
    Behaviour: parses standard columns; lines that do not parse are skipped.
    """
    out = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            out.append({"rec": line[:6].strip(), "resname": line[17:20].strip(),
                        "name": line[12:16].strip(), "chain": line[21:22].strip(),
                        "resseq": line[22:26].strip(),
                        "xyz": np.array([x, y, z])})
    return out


def resolve_seed(original_pdb, fixed_pdb, opts):
    """
    Choose the site seed point by the §10 priority order.

    Input:  original_pdb - the uploaded PDB (holds any native ligand HETATM)
            fixed_pdb     - the PDBFixer output (for residue-mask centroids)
            opts          - job options dict; keys read:
                            center_x/center_y/center_z - explicit seed
                            site_residues - list of "CHAIN:RESSEQ" or RESSEQ
            native_resname (opt) - force a specific HETATM resname as native
    Output: (np.array([x,y,z]), source_label)
    Behaviour: explicit XYZ wins; else the centroid of the named residues in the
               fixed structure; else the centroid of the largest non-solvent
               HETATM group in the original PDB. Raises ValueError if none apply.
    """
    cx, cy, cz = opts.get("center_x"), opts.get("center_y"), opts.get("center_z")
    if cx is not None and cy is not None and cz is not None:
        return np.array([float(cx), float(cy), float(cz)]), "explicit"

    mask = opts.get("site_residues")
    if mask:
        # entries are "CHAIN:RESSEQ" or bare "RESSEQ"; match on resSeq, and on
        # chain too when the entry gives one.
        want = set()
        for m in mask:
            s = str(m)
            want.add((s.split(":")[0], s.split(":")[1]) if ":" in s
                     else (None, s))
        pts = [a["xyz"] for a in _read_atoms(fixed_pdb)
               if (a["chain"], a["resseq"]) in want or (None, a["resseq"]) in want]
        if pts:
            return np.mean(pts, axis=0), "residue-mask"

    # native ligand: largest non-solvent HETATM residue group
    force = opts.get("native_resname")
    groups = {}
    for a in _read_atoms(original_pdb):
        if a["rec"] != "HETATM":
            continue
        if force:
            if a["resname"] != force:
                continue
        elif a["resname"].upper() in _SKIP_HET:
            continue
        groups.setdefault((a["resname"], a["chain"], a["resseq"]),
                          []).append(a["xyz"])
    if groups:
        biggest = max(groups.values(), key=len)
        return np.mean(biggest, axis=0), "native-ligand"

    raise ValueError("no seed: provide center_x/y/z, site_residues, or a "
                     "native ligand HETATM in the PDB")


# ══════════════════════════════════════════════════════════════════════════════
# Box, reorientation, core mask
# ══════════════════════════════════════════════════════════════════════════════

def box_from_points(points, pad=4.0):
    """
    Axis-aligned docking box enclosing a point cloud, with padding.

    Input:  points - (N,3) coordinates (already in the target frame)
            pad    - angstrom added on every side
    Output: dict {cx,cy,cz,sx,sy,sz}
    """
    lo = points.min(axis=0) - pad
    hi = points.max(axis=0) + pad
    cen = (lo + hi) / 2.0
    size = hi - lo
    return {"cx": round(float(cen[0]), 3), "cy": round(float(cen[1]), 3),
            "cz": round(float(cen[2]), 3), "sx": round(float(size[0]), 3),
            "sy": round(float(size[1]), 3), "sz": round(float(size[2]), 3)}


def _box_volume(points, pad):
    lo = points.min(axis=0) - pad
    hi = points.max(axis=0) + pad
    return float(np.prod(hi - lo))


def reorient_decision(points, pad=4.0, gain_threshold=2.0, seed=0):
    """
    Decide whether reorienting the region is worth it, and if so by what rotation.

    Input:  points        - (N,3) region coordinates in the original frame
            pad           - box padding used for the volume comparison
            gain_threshold- reorient only if axis-aligned box volume shrinks by
                            at least this factor
            seed          - RNG seed for best_orientation
    Output: (R or None, transform 4x4, oriented_points, gain)
            R is None when the original frame is kept (transform = identity).
    Behaviour: compares the as-is axis-aligned box volume to the best-oriented
               one (box_check.best_orientation, rotating about the centroid). A
               rotation makes every stored coordinate frame-dependent, so it is
               only taken when it pays off by at least gain_threshold.
    """
    c = points.mean(axis=0)
    vol_asis = _box_volume(points, pad)
    R, _, _ = bc.best_orientation(points, pad=pad, seed=seed)
    oriented = (points - c) @ R + c
    vol_rot = _box_volume(oriented, pad)
    gain = vol_asis / vol_rot if vol_rot > 0 else 1.0

    if gain >= gain_threshold:
        T = np.eye(4)
        T[:3, :3] = R.T                       # x' = (x-c) @ R + c  == R.T (x-c) + c
        T[:3, 3] = c - R.T @ c
        return R, T, oriented, round(float(gain), 3)
    return None, np.eye(4), points.copy(), round(float(gain), 3)


def apply_transform(points, T):
    """Apply a 4x4 homogeneous transform to (N,3) points."""
    P = np.hstack([points, np.ones((len(points), 1))])
    return (P @ T.T)[:, :3]


def rasterize_core(core_points, spacing, box):
    """
    Rasterise the pocket-core points into a boolean grid over the final box and
    run-length-encode the occupied cells.

    Input:  core_points - (N,3) core coordinates IN THE FINAL (box) frame
            spacing     - grid spacing (angstrom)
            box         - the final box dict {cx,cy,cz,sx,sy,sz}
    Output: dict {shape:[nz,ny,nx], spacing, origin:[x0,y0,z0], runs:[[start,len]]}
            where runs are over the C-order (z,y,x) flattened occupancy grid.
    Behaviour: cells containing a core point are marked occupied; the mask is
               stored compactly so the docking engine can test pose occupancy of
               the pocket without the apron.
    """
    nx = max(1, int(round(box["sx"] / spacing)))
    ny = max(1, int(round(box["sy"] / spacing)))
    nz = max(1, int(round(box["sz"] / spacing)))
    origin = np.array([box["cx"] - box["sx"] / 2.0,
                       box["cy"] - box["sy"] / 2.0,
                       box["cz"] - box["sz"] / 2.0])
    grid = np.zeros((nz, ny, nx), dtype=bool)
    idx = np.floor((core_points - origin) / spacing).astype(int)
    for ix, iy, iz in idx:
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            grid[iz, iy, ix] = True
    flat = grid.reshape(-1)
    runs = []
    i = 0
    n = flat.size
    while i < n:
        if flat[i]:
            j = i
            while j < n and flat[j]:
                j += 1
            runs.append([int(i), int(j - i)])
            i = j
        else:
            i += 1
    return {"shape": [nz, ny, nx], "spacing": float(spacing),
            "origin": [round(float(v), 4) for v in origin], "runs": runs}


def native_ligand_atoms(original_pdb, native_resname=None):
    """
    Heavy-atom coordinates of the native ligand, if the PDB has one.

    Input:  original_pdb   - the uploaded PDB
            native_resname - force a specific HETATM resname; otherwise the
                             largest non-solvent HETATM group is used
    Output: (N,3) array of the native ligand's atoms, or an empty (0,3) array
    Behaviour: mirrors resolve_seed's native-ligand selection so the validated
               ligand is the same one that would have seeded the search.
    """
    groups = {}
    for a in _read_atoms(original_pdb):
        if a["rec"] != "HETATM":
            continue
        if native_resname:
            if a["resname"] != native_resname:
                continue
        elif a["resname"].upper() in _SKIP_HET:
            continue
        groups.setdefault((a["resname"], a["chain"], a["resseq"]), []).append(a["xyz"])
    if not groups:
        return np.zeros((0, 3))
    return np.array(max(groups.values(), key=len))


def fraction_in_core(points, core_mask):
    """
    Fraction of points that fall inside the stored pocket core.

    Input:  points    - (N,3) coordinates IN THE FINAL (box) frame
            core_mask - dict from rasterize_core
    Output: float in [0,1]; 0.0 for empty input
    Behaviour: used at prep to confirm a native ligand sits in the grown core;
               a low fraction means the burial threshold or the structure is
               wrong and the box should not be trusted.
    """
    pts = np.asarray(points, float)
    if len(pts) == 0:
        return 0.0
    return sum(1 for p in pts if core_contains(p, core_mask)) / len(pts)


def core_contains(xyz, core_mask):
    """
    Dock-time test: is a point inside the stored pocket core?

    Input:  xyz       - a point in the FINAL (box) frame
            core_mask - the dict returned by rasterize_core
    Output: bool
    Behaviour: maps the point to its grid cell and checks the run-length table.
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


# ══════════════════════════════════════════════════════════════════════════════
# Measurement (given a probe map array)
# ══════════════════════════════════════════════════════════════════════════════

def measure_from_map(arr, spacing, center, npts, seed_xyz,
                     apron_reach=8.0, reorient_gain=2.0, pad=4.0,
                     threshold=1.0, burial_radius=6.0):
    """
    Measure the box, transform and core mask from a probe map array.

    Input:  arr, spacing, center, npts - an AutoGrid probe map (box_check.read_map)
            seed_xyz      - site seed point
            apron_reach   - surface apron reach (0 disables the apron)
            reorient_gain - reorient only if box volume shrinks by >= this factor
            pad           - box padding
    Output: dict {box, transform (4x4 list), core_mask, seed, source_volumes,
                  reoriented, gain, touches_edge}
    Behaviour: grows the pocket core (and apron, if enabled) with box_check,
               decides reorientation by volume gain, emits the box in the chosen
               frame, and run-length-encodes the core in that frame. Raises
               ValueError if box_check cannot grow a confined cavity.
    """
    core = bc.grow_site_with_apron(arr, spacing, center, npts, seed_xyz,
                                   threshold=threshold,
                                   burial_radius=burial_radius,
                                   apron_reach=0.0)
    if "error" in core:
        raise ValueError(f"cavity growth failed: {core['error']}")
    core_points = np.asarray(core["region_points"], dtype=float)

    if apron_reach and apron_reach > 0:
        full = bc.grow_site_with_apron(arr, spacing, center, npts, seed_xyz,
                                       threshold=threshold,
                                       burial_radius=burial_radius,
                                       apron_reach=apron_reach)
        region_points = np.asarray(full["region_points"], dtype=float)
        touches_edge = bool(full.get("touches_edge", False))
    else:
        region_points = core_points
        touches_edge = bool(core.get("touches_edge", False))

    R, T, oriented, gain = reorient_decision(region_points, pad, reorient_gain)
    box = box_from_points(oriented, pad)
    core_in_frame = apply_transform(core_points, T) if R is not None else core_points
    core_mask = rasterize_core(core_in_frame, spacing, box)

    return {
        "box": box,
        "transform": [[round(float(v), 6) for v in row] for row in T],
        "core_mask": core_mask,
        "seed": [round(float(v), 3) for v in seed_xyz],
        "reoriented": R is not None,
        "gain": gain,
        "core_volume": round(float(len(core_points)) * spacing ** 3, 1),
        "region_volume": round(float(len(region_points)) * spacing ** 3, 1),
        "touches_edge": touches_edge,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Provisional grid (container-only: runs autogrid4 via an injected callable)
# ══════════════════════════════════════════════════════════════════════════════

def measure_box(rigid_pdbqt, seed_xyz, spacing, generate_maps_fn, workdir,
                provisional_size=30.0, probe="C", apron_reach=8.0,
                reorient_gain=2.0, pad=4.0):
    """
    Full measurement: build a throwaway probe grid around the seed, then measure.

    Input:  rigid_pdbqt     - prepared rigid receptor PDBQT
            seed_xyz        - site seed point
            spacing         - grid spacing
            generate_maps_fn- callable(rigid_pdbqt, maps_dir, box, spacing,
                              [probe]) -> fld_path (prep_receptor.generate_maps)
            workdir         - scratch directory for the provisional maps
            provisional_size- edge of the cubic provisional box around the seed
    Output: the dict from measure_from_map
    Behaviour: the provisional grid exists only to compute burial on and is
               discarded; it is centred on the seed and sized to enclose the
               pocket plus enough protein context for the burial test. Requires
               autogrid4 (via generate_maps_fn) and so runs only in the container.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    prov_box = {"cx": float(seed_xyz[0]), "cy": float(seed_xyz[1]),
                "cz": float(seed_xyz[2]), "sx": provisional_size,
                "sy": provisional_size, "sz": provisional_size}
    maps_dir = workdir / "prov_maps"
    generate_maps_fn(rigid_pdbqt, maps_dir, prov_box, spacing, [probe])
    arr, sp, npts, center = bc.read_map(maps_dir / f"receptor.{probe}.map")
    return measure_from_map(arr, sp, center, npts, np.asarray(seed_xyz, float),
                            apron_reach, reorient_gain, pad)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(prog="cavity_box.py", add_help=False,
        description="Measure a docking box from a receptor cavity.")
    p.add_argument("--map", metavar="PATH", help="A single AutoGrid probe map")
    p.add_argument("--seed", metavar="X,Y,Z", help="Seed point or 'center'")
    p.add_argument("--pad", type=float, default=4.0, metavar="AA")
    p.add_argument("--apron-reach", type=float, default=8.0, metavar="AA")
    p.add_argument("--reorient-gain", type=float, default=2.0, metavar="G")
    p.add_argument("--json", metavar="PATH")
    p.add_argument("--help", action="help", help="Show this message")
    args = p.parse_args()

    if not args.map or not args.seed:
        print("ERROR: --map and --seed are required", file=sys.stderr)
        sys.exit(2)
    arr, sp, npts, center = bc.read_map(Path(args.map))
    if args.seed.strip().lower() == "center":
        seed = np.asarray(center if center else (0, 0, 0), float)
    else:
        seed = np.asarray([float(v) for v in args.seed.replace(",", " ").split()],
                          float)
    res = measure_from_map(arr, sp, center, npts, seed, args.apron_reach,
                           args.reorient_gain, args.pad)
    b = res["box"]
    print(f"  seed        : {res['seed']}")
    print(f"  box centre  : ({b['cx']}, {b['cy']}, {b['cz']})")
    print(f"  box size    : {b['sx']} x {b['sy']} x {b['sz']} A")
    print(f"  reoriented  : {res['reoriented']} (volume gain {res['gain']}x)")
    print(f"  core volume : {res['core_volume']} A^3   "
          f"region {res['region_volume']} A^3")
    if res["touches_edge"]:
        print("  WARNING: region reaches the provisional map edge")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
