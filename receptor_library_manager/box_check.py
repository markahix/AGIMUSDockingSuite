#!/usr/bin/env python3
"""
box_check.py — does the search box have room for the ligands?

A docking box can be large enough to contain a ligand's bounding dimensions
and still have nowhere for it to go, because most of the box is occupied by
protein. When that happens AutoDock does not fail: it returns the best pose
it can find, which is typically draped across the receptor surface with a
plausible-looking score. This tool makes that condition visible before a run
rather than after.

Two quantities are compared:

  required   The smallest volume a ligand can occupy in any conformation,
             from the union of its van der Waals spheres divided by the
             close-packing fraction. This is a hard floor: no conformer,
             however folded, occupies less. Note it is the UNION volume,
             not the sum of atomic volumes, which overcounts by ~1.7x
             because bonded atoms interpenetrate.

  available  Accessible volume inside the box, counted from an AutoGrid
             map: grid points whose interaction energy is below a threshold
             are free space. Reported both as the total and as the largest
             contiguous pocket, since a ligand needs connected space and
             total free volume may be split across disjoint cavities.

Usage:
    python3 box_check.py --maps MAPS_DIR --library LIB_DIR [options]
    python3 box_check.py --maps MAPS_DIR --ligand LIGAND.pdb

Options:
    --maps DIR          Directory holding receptor.*.map (a model's maps/ dir)
    --library DIR       Library directory; checks every <name>/<name>.pdb
    --ligand PATH       Check a single ligand PDB instead
    --map-type TYPE     Probe map to use for accessibility (default: C)
    --threshold KCAL    Energy below which a grid point is free (default: 1.0)
    --packing FRAC      Close-packing fraction (default: 0.74)
    --samples N         Monte Carlo samples for union volume (default: 300000)
    --json PATH         Write results as JSON
    --help              Show this message
"""

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

_VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
        "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
        "MG": 1.73, "CA": 2.31, "ZN": 1.39, "FE": 2.00, "MN": 2.00}
_CLOSE_PACKING = 0.74


# ══════════════════════════════════════════════════════════════════════════════
# AutoGrid map parsing
# ══════════════════════════════════════════════════════════════════════════════

def read_map(map_path):
    """
    Parse an AutoGrid .map file.

    Header is six keyword lines, then one energy value per line ordered with
    x fastest, then y, then z. NELEMENTS n means n+1 points along that axis.
    Returns (values, spacing, npts, center) with values shaped (nz, ny, nx).
    """
    spacing = None
    npts = None
    center = None
    values = []
    with open(map_path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            key = s.split()[0].upper()
            if key == "SPACING":
                spacing = float(s.split()[1])
            elif key == "NELEMENTS":
                npts = tuple(int(x) + 1 for x in s.split()[1:4])
            elif key == "CENTER":
                center = tuple(float(x) for x in s.split()[1:4])
            elif key in ("GRID_PARAMETER_FILE", "GRID_DATA_FILE", "MACROMOLECULE"):
                continue
            else:
                try:
                    values.append(float(s))
                except ValueError:
                    continue
    if spacing is None or npts is None:
        raise ValueError(f"{map_path}: incomplete header")
    nx, ny, nz = npts
    expect = nx * ny * nz
    if len(values) != expect:
        raise ValueError(f"{map_path}: expected {expect} values, read {len(values)}")
    arr = np.asarray(values, dtype=np.float32).reshape((nz, ny, nx))
    return arr, spacing, npts, center


def find_pockets(mask, spacing, center, npts):
    """
    Every 6-connected component of the accessibility mask, with its volume
    and Cartesian centroid.

    Reporting only the largest component answers "does anything fit" but not
    "where should the box be centred", and it hides a second cavity that is
    also large enough. Both questions matter when the verdict is that the
    current box is unusable.
    """
    nz, ny, nx = mask.shape
    cell = spacing ** 3
    cx, cy, cz = center if center else (0.0, 0.0, 0.0)

    def to_cart(ix, iy, iz):
        return (cx + (ix - (nx - 1) / 2.0) * spacing,
                cy + (iy - (ny - 1) / 2.0) * spacing,
                cz + (iz - (nz - 1) / 2.0) * spacing)

    labels = None
    try:
        from scipy import ndimage           # optional; absent in lean images
        labels, ncomp = ndimage.label(mask)
    except Exception:
        labels = np.zeros(mask.shape, dtype=np.int32)
        ncomp = 0
        for start in np.argwhere(mask):
            z0, y0, x0 = (int(v) for v in start)
            if labels[z0, y0, x0]:
                continue
            ncomp += 1
            dq = deque([(z0, y0, x0)])
            labels[z0, y0, x0] = ncomp
            while dq:
                z, y, x = dq.popleft()
                for dz, dy, dx in ((1,0,0),(-1,0,0),(0,1,0),
                                   (0,-1,0),(0,0,1),(0,0,-1)):
                    a, b, c = z+dz, y+dy, x+dx
                    if 0 <= a < nz and 0 <= b < ny and 0 <= c < nx:
                        if mask[a, b, c] and not labels[a, b, c]:
                            labels[a, b, c] = ncomp
                            dq.append((a, b, c))

    pockets = []
    for lab in range(1, ncomp + 1):
        idx = np.argwhere(labels == lab)
        if idx.size == 0:
            continue
        zc, yc, xc = idx.mean(axis=0)
        pockets.append({
            "volume":   round(float(len(idx) * cell), 1),
            "n_points": int(len(idx)),
            "centroid": [round(float(v), 2) for v in to_cart(xc, yc, zc)],
        })
    pockets.sort(key=lambda d: -d["volume"])
    return pockets


# ══════════════════════════════════════════════════════════════════════════════
# Seeded pocket growth
# ══════════════════════════════════════════════════════════════════════════════

def burial_fraction(occupied, radius_pts):
    """
    Fraction of a surrounding cube that is protein-occupied, per grid point.

    Computed from a 3D summed-area table so each point costs O(1) regardless
    of radius. This separates a buried pocket from bulk solvent: a point deep
    in a cleft is enclosed on most sides, a point in open solvent is not.
    Without this filter a seeded flood fill escapes any surface-accessible
    pocket and reports the volume of everything outside the protein.
    """
    occ = occupied.astype(np.float32)
    S = occ.cumsum(0).cumsum(1).cumsum(2)
    S = np.pad(S, ((1, 0), (1, 0), (1, 0)), mode="constant")
    nz, ny, nx = occ.shape
    r = int(max(1, radius_pts))

    z0 = np.clip(np.arange(nz) - r, 0, nz); z1 = np.clip(np.arange(nz) + r + 1, 0, nz)
    y0 = np.clip(np.arange(ny) - r, 0, ny); y1 = np.clip(np.arange(ny) + r + 1, 0, ny)
    x0 = np.clip(np.arange(nx) - r, 0, nx); x1 = np.clip(np.arange(nx) + r + 1, 0, nx)

    Z0, Y0, X0 = np.meshgrid(z0, y0, x0, indexing="ij")
    Z1, Y1, X1 = np.meshgrid(z1, y1, x1, indexing="ij")

    total = (S[Z1, Y1, X1] - S[Z0, Y1, X1] - S[Z1, Y0, X1] - S[Z1, Y1, X0]
             + S[Z0, Y0, X1] + S[Z0, Y1, X0] + S[Z1, Y0, X0] - S[Z0, Y0, X0])
    count = ((Z1 - Z0) * (Y1 - Y0) * (X1 - X0)).astype(np.float32)
    return total / np.maximum(count, 1.0)


def grow_from_seed(arr, spacing, center, npts, seed_xyz,
                   threshold=1.0, burial_radius=6.0, burial_min=0.45):
    """
    Grow the cavity containing a known site point.

    Rather than taking whichever cavity is largest, this starts at a point
    the caller knows to be the active site -- the native ligand centroid,
    say -- and expands outward through connected free space that is
    sufficiently buried. The resulting extent is a direct measurement of the
    pocket, so the box can be sized to the site instead of guessed from
    ligand dimensions plus padding.

    Reports whether the grown region reaches the edge of the map, which
    means the box is cutting the pocket short.
    """
    nz, ny, nx = arr.shape
    cx, cy, cz = center if center else (0.0, 0.0, 0.0)

    def to_idx(x, y, z):
        return (int(round((x - cx) / spacing + (nx - 1) / 2.0)),
                int(round((y - cy) / spacing + (ny - 1) / 2.0)),
                int(round((z - cz) / spacing + (nz - 1) / 2.0)))

    def to_cart(ix, iy, iz):
        return (cx + (ix - (nx - 1) / 2.0) * spacing,
                cy + (iy - (ny - 1) / 2.0) * spacing,
                cz + (iz - (nz - 1) / 2.0) * spacing)

    free     = arr < threshold
    occupied = ~free
    burial   = burial_fraction(occupied, burial_radius / spacing)
    pocket   = free & (burial >= burial_min)

    ix, iy, iz = to_idx(*seed_xyz)
    if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
        return {"error": "seed lies outside the map"}

    seed_state = ("free" if free[iz, iy, ix] else "inside protein")
    # Nearest qualifying point to the seed, if the seed itself does not qualify.
    moved = 0.0
    if not pocket[iz, iy, ix]:
        cand = np.argwhere(pocket)
        if cand.size == 0:
            return {"error": "no buried free space in this map at the given "
                             "burial threshold"}
        d = ((cand - np.array([iz, iy, ix])) ** 2).sum(axis=1)
        iz, iy, ix = (int(v) for v in cand[int(d.argmin())])
        moved = float(np.sqrt(d.min())) * spacing

    visited = np.zeros(arr.shape, dtype=bool)
    visited[iz, iy, ix] = True
    dq = deque([(iz, iy, ix)])
    pts = []
    touches_edge = False
    while dq:
        z, y, x = dq.popleft()
        pts.append((x, y, z))
        if x in (0, nx - 1) or y in (0, ny - 1) or z in (0, nz - 1):
            touches_edge = True
        for dz, dy, dx in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            a, b, c = z+dz, y+dy, x+dx
            if 0 <= a < nz and 0 <= b < ny and 0 <= c < nx:
                if pocket[a, b, c] and not visited[a, b, c]:
                    visited[a, b, c] = True
                    dq.append((a, b, c))

    P = np.array(pts)
    lo = P.min(axis=0); hi = P.max(axis=0)
    lo_c = to_cart(*lo); hi_c = to_cart(*hi)
    cen_c = to_cart(*P.mean(axis=0))
    return {
        "seed":          [round(float(v), 2) for v in seed_xyz],
        "seed_state":    seed_state,
        "seed_moved":    round(moved, 2),
        "volume":        round(len(pts) * spacing ** 3, 1),
        "centroid":      [round(float(v), 2) for v in cen_c],
        "extent":        [round(float(hi_c[i] - lo_c[i]), 2) for i in range(3)],
        "bbox_min":      [round(float(v), 2) for v in lo_c],
        "bbox_max":      [round(float(v), 2) for v in hi_c],
        "touches_edge":  bool(touches_edge),
        "burial_min":    burial_min,
        "burial_radius": burial_radius,
    }


def grow_site(arr, spacing, center, npts, seed_xyz, threshold=1.0,
              burial_radius=6.0, burial_levels=(0.45, 0.55, 0.65, 0.75, 0.85)):
    """
    Grow the site cavity, escalating the burial threshold until the region
    stops reaching the edge of the map.

    A region that touches the edge is ambiguous: either the box is cutting
    the pocket short, or the fill has leaked into bulk solvent. Both make
    the measured volume meaningless, and which one it is cannot be told from
    inside the box. Escalating the threshold resolves the leak case; if the
    region still reaches the edge at the strictest level, the box genuinely
    is too small and the result says so rather than returning a number that
    looks usable.
    """
    attempts = []
    for b in burial_levels:
        r = grow_from_seed(arr, spacing, center, npts, seed_xyz,
                           threshold, burial_radius, b)
        if "error" in r:
            attempts.append({"burial_min": b, "error": r["error"]})
            continue
        attempts.append({"burial_min": b, "volume": r["volume"],
                         "touches_edge": r["touches_edge"]})
        if not r["touches_edge"]:
            r["attempts"] = attempts
            r["confined"] = True
            return r
    last = grow_from_seed(arr, spacing, center, npts, seed_xyz,
                          threshold, burial_radius, burial_levels[-1])
    if "error" not in last:
        last["attempts"] = attempts
        last["confined"] = False
    return last


def _dilate(mask, steps):
    """6-connected dilation, `steps` iterations, numpy shifts only."""
    out = mask.copy()
    for _ in range(int(steps)):
        nxt = out.copy()
        nxt[1:, :, :] |= out[:-1, :, :]
        nxt[:-1, :, :] |= out[1:, :, :]
        nxt[:, 1:, :] |= out[:, :-1, :]
        nxt[:, :-1, :] |= out[:, 1:, :]
        nxt[:, :, 1:] |= out[:, :, :-1]
        nxt[:, :, :-1] |= out[:, :, 1:]
        out = nxt
    return out


def grow_site_with_apron(arr, spacing, center, npts, seed_xyz,
                         threshold=1.0, burial_radius=6.0,
                         core_burial=None, apron_burial=0.35,
                         apron_reach=8.0):
    """
    Grow the site cavity, then extend onto the adjacent protein surface.

    Ligands anchored in a cleft frequently place functional groups outside
    it, flat against the surface, where they pick up additional contacts.
    A pocket-only mask deletes precisely the region those groups occupy, so
    the search space has to include an apron of surface-adjacent space.

    The apron is defined as free space within `apron_reach` of the pocket
    core whose burial still exceeds `apron_burial`. The burial floor is what
    separates surface from bulk: a point lying against the protein face is
    enclosed on one side, a point in open solvent on none.

    core_burial defaults to whatever level grow_site() needed to confine the
    core, so the apron is always measured relative to a core that did not leak.
    """
    core = (grow_site(arr, spacing, center, npts, seed_xyz,
                      threshold, burial_radius)
            if core_burial is None else
            grow_from_seed(arr, spacing, center, npts, seed_xyz,
                           threshold, burial_radius, core_burial))
    if "error" in core:
        return core

    nz, ny, nx = arr.shape
    cx, cy, cz = center if center else (0.0, 0.0, 0.0)

    def to_idx(x, y, z):
        return (int(round((x - cx) / spacing + (nx - 1) / 2.0)),
                int(round((y - cy) / spacing + (ny - 1) / 2.0)),
                int(round((z - cz) / spacing + (nz - 1) / 2.0)))

    def to_cart(ix, iy, iz):
        return (cx + (ix - (nx - 1) / 2.0) * spacing,
                cy + (iy - (ny - 1) / 2.0) * spacing,
                cz + (iz - (nz - 1) / 2.0) * spacing)

    free   = arr < threshold
    burial = burial_fraction(~free, burial_radius / spacing)

    # Rebuild the core mask by refilling at the burial level it settled on.
    core_mask = np.zeros(arr.shape, dtype=bool)
    pocket = free & (burial >= core["burial_min"])
    ix, iy, iz = to_idx(*core["centroid"])
    if not pocket[iz, iy, ix]:
        cand = np.argwhere(pocket)
        d = ((cand - np.array([iz, iy, ix])) ** 2).sum(axis=1)
        iz, iy, ix = (int(v) for v in cand[int(d.argmin())])
    dq = deque([(iz, iy, ix)])
    core_mask[iz, iy, ix] = True
    while dq:
        z, y, x = dq.popleft()
        for dz, dy, dx in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            a, b, c = z+dz, y+dy, x+dx
            if 0 <= a < nz and 0 <= b < ny and 0 <= c < nx:
                if pocket[a, b, c] and not core_mask[a, b, c]:
                    core_mask[a, b, c] = True
                    dq.append((a, b, c))

    reached = _dilate(core_mask, round(apron_reach / spacing))
    apron   = reached & free & (burial >= apron_burial) & ~core_mask
    region  = core_mask | apron

    cell = spacing ** 3
    P = np.argwhere(region)                       # (z, y, x)
    pts = np.stack([P[:, 2], P[:, 1], P[:, 0]], axis=1).astype(float)
    cart = np.array([to_cart(*q) for q in pts])

    return {
        "core":           core,
        "core_volume":    round(float(core_mask.sum()) * cell, 1),
        "apron_volume":   round(float(apron.sum()) * cell, 1),
        "region_volume":  round(float(region.sum()) * cell, 1),
        "apron_burial":   apron_burial,
        "apron_reach":    apron_reach,
        "region_points":  cart,
        "region_centroid": [round(float(v), 2) for v in cart.mean(axis=0)],
        "region_extent":  [round(float(v), 2) for v in
                           (cart.max(axis=0) - cart.min(axis=0))],
        "touches_edge":   bool(region[0].any() or region[-1].any() or
                               region[:, 0].any() or region[:, -1].any() or
                               region[:, :, 0].any() or region[:, :, -1].any()),
    }


def best_orientation(points, pad=4.0, n_trials=3000, seed=0, refine=True):
    """
    Rotation minimising the axis-aligned bounding box of `points`.

    PCA is not used: it maximises variance along the first axis, which
    elongates the box rather than compacting it, and measured on real
    geometry it lands near the 80th percentile of random orientations --
    worse than leaving the structure alone. A direct search over rotations
    optimises the quantity that matters and costs milliseconds.

    Returns (R, dims, volume) with R applied as  points_rotated = points @ R.
    """
    rng = np.random.default_rng(seed)
    P = points - points.mean(axis=0)

    def vol_of(R):
        Q = P @ R
        e = Q.max(axis=0) - Q.min(axis=0) + 2 * pad
        return float(np.prod(e)), e

    bestR = np.eye(3)
    bestV, bestE = vol_of(bestR)
    for _ in range(int(n_trials)):
        A = rng.normal(size=(3, 3))
        R, _ = np.linalg.qr(A)
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1.0
        v, e = vol_of(R)
        if v < bestV:
            bestV, bestE, bestR = v, e, R

    if refine:                                   # local polish about the best
        step = 0.25
        for _ in range(6):
            improved = False
            for axis in range(3):
                for sgn in (1.0, -1.0):
                    ang = sgn * step
                    c, s = math.cos(ang), math.sin(ang)
                    K = np.eye(3)
                    a, b = [i for i in range(3) if i != axis]
                    K[a, a] = c; K[a, b] = -s; K[b, a] = s; K[b, b] = c
                    cand = bestR @ K
                    v, e = vol_of(cand)
                    if v < bestV:
                        bestV, bestE, bestR = v, e, cand
                        improved = True
            if not improved:
                step *= 0.5
    return bestR, bestE, bestV


# ══════════════════════════════════════════════════════════════════════════════
# Ligand volume
# ══════════════════════════════════════════════════════════════════════════════

def read_heavy(pdb_path):
    elems, xyz = [], []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            name = line[12:16].strip()
            el = (line[76:78].strip() or "").upper()
            if not el:
                raw = "".join(c for c in name if c.isalpha())
                el = raw[:1].upper()
            if el in ("H", "D"):
                continue
            try:
                xyz.append([float(line[30:38]), float(line[38:46]),
                            float(line[46:54])])
            except ValueError:
                continue
            elems.append(el)
    return elems, np.asarray(xyz, dtype=float)


def union_volume(elems, xyz, samples=300000, seed=0):
    """
    Monte Carlo volume of the union of van der Waals spheres.

    The union is required rather than the sum: bonded atoms overlap heavily,
    and summing atomic volumes overstates the true occupied volume by roughly
    1.7x for a typical drug-like molecule, which would reject boxes that are
    in fact large enough.
    """
    if len(xyz) == 0:
        return 0.0
    r = np.array([_VDW.get(e, 1.7) for e in elems])
    lo = (xyz - r[:, None]).min(axis=0)
    hi = (xyz + r[:, None]).max(axis=0)
    box = float(np.prod(hi - lo))
    rng = np.random.default_rng(seed)
    pts = rng.uniform(lo, hi, size=(samples, 3))
    inside = np.zeros(samples, dtype=bool)
    for k in range(len(xyz)):
        inside |= ((pts - xyz[k]) ** 2).sum(axis=1) < r[k] ** 2
    return box * float(inside.mean())


def required_volume(elems, xyz, packing=_CLOSE_PACKING, samples=300000):
    """Hard floor on the volume any conformation of this ligand occupies."""
    return union_volume(elems, xyz, samples) / packing


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

def analyse_box(maps_dir, map_type="C", threshold=1.0):
    maps_dir = Path(maps_dir)
    cand = maps_dir / f"receptor.{map_type}.map"
    if not cand.exists():
        found = sorted(maps_dir.glob("receptor.*.map"))
        if not found:
            raise FileNotFoundError(f"no receptor.*.map in {maps_dir}")
        cand = found[0]
    arr, spacing, npts, center = read_map(cand)
    cell = spacing ** 3
    mask = arr < threshold
    total_free = int(mask.sum())
    pockets = find_pockets(mask, spacing, center, npts)
    pocket  = pockets[0]["volume"] if pockets else 0.0
    ncomp   = len(pockets)
    nx, ny, nz = npts
    return {
        "map_file":      cand.name,
        "spacing":       spacing,
        "npts":          list(npts),
        "center":        list(center) if center else None,
        "box_dims":      [round((n - 1) * spacing, 2) for n in (nx, ny, nz)],
        "box_volume":    round((nx - 1) * (ny - 1) * (nz - 1) * cell, 1),
        "free_volume":   round(total_free * cell, 1),
        "pocket_volume": pocket,
        "n_pockets":     ncomp,
        "pockets":       pockets,
        "occupancy":     round(1.0 - total_free / max(arr.size, 1), 3),
    }


def main():
    p = argparse.ArgumentParser(
        prog="box_check.py",
        description="Check whether a docking box has room for the ligands.")
    p.add_argument("--maps", required=True, metavar="DIR",
                   help="Directory holding receptor.*.map")
    p.add_argument("--library", metavar="DIR",
                   help="Library directory; checks every <name>/<name>.pdb")
    p.add_argument("--ligand", metavar="PATH", help="Check a single ligand PDB")
    p.add_argument("--map-type", default="C", metavar="TYPE",
                   help="Probe map used for accessibility (default: C)")
    p.add_argument("--threshold", type=float, default=1.0, metavar="KCAL",
                   help="Energy below which a grid point is free (default: 1.0)")
    p.add_argument("--packing", type=float, default=_CLOSE_PACKING, metavar="FRAC",
                   help="Close-packing fraction (default: 0.74)")
    p.add_argument("--samples", type=int, default=300000, metavar="N",
                   help="Monte Carlo samples for union volume (default: 300000)")
    p.add_argument("--seed", metavar="X,Y,Z",
                   help="Grow the cavity containing this point (e.g. the native "
                        "ligand centroid) instead of reporting the largest. "
                        "Use 'center' for the box centre")
    p.add_argument("--burial-radius", type=float, default=6.0, metavar="AA",
                   help="Neighbourhood radius for the burial test (default: 6)")
    p.add_argument("--json", metavar="PATH", help="Write results as JSON")
    args = p.parse_args()

    if not args.library and not args.ligand:
        print("ERROR: give --library or --ligand", file=sys.stderr)
        sys.exit(1)

    try:
        box = analyse_box(args.maps, args.map_type, args.threshold)
    except Exception as exc:
        print(f"ERROR reading maps: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.seed:
        from pathlib import Path as _P
        mp = _P(args.maps) / box["map_file"]
        arr, sp, npts, cen = read_map(mp)
        if args.seed.strip().lower() == "center":
            seed = cen if cen else (0.0, 0.0, 0.0)
        else:
            try:
                seed = tuple(float(v) for v in args.seed.replace(",", " ").split())
            except ValueError:
                print("ERROR: --seed must be X,Y,Z or 'center'", file=sys.stderr)
                sys.exit(1)
        site = grow_site(arr, sp, cen, npts, seed, args.threshold,
                         args.burial_radius)
        if "error" in site:
            print(f"ERROR: {site['error']}", file=sys.stderr)
            sys.exit(1)
        print(f"  Seeded site growth")
        print(f"    seed          : {site['seed']}  ({site['seed_state']})")
        if site["seed_moved"]:
            print(f"    seed relocated: {site['seed_moved']} A to nearest "
                  f"buried free point")
        print(f"    burial used   : {site['burial_min']:.2f} "
              f"(radius {site['burial_radius']} A)")
        print(f"    volume        : {site['volume']:.0f} A^3")
        print(f"    centroid      : {site['centroid']}")
        print(f"    extent        : {site['extent'][0]} x {site['extent'][1]} "
              f"x {site['extent'][2]} A")
        if not site.get("confined", True):
            print(f"    WARNING: region still reaches the map edge at the "
                  f"strictest burial level.")
            print(f"             The box is truncating this pocket; re-prep "
                  f"larger, centred on {site['centroid']}.")
        else:
            print(f"    fully enclosed by the box: yes")
        print()
        box["seeded_site"] = site

    print(f"  Map            : {box['map_file']}  spacing {box['spacing']} A")
    print(f"  Box            : {box['box_dims'][0]} x {box['box_dims'][1]} "
          f"x {box['box_dims'][2]} A   = {box['box_volume']:.0f} A^3")
    print(f"  Free volume    : {box['free_volume']:.0f} A^3 "
          f"({100*(1-box['occupancy']):.0f}% of box)")
    print(f"  Largest pocket : {box['pocket_volume']:.0f} A^3 "
          f"across {box['n_pockets']} disconnected region(s)")
    print()

    ligs = []
    if args.ligand:
        ligs = [Path(args.ligand)]
    else:
        lib = Path(args.library)
        for sub in sorted(x for x in lib.iterdir() if x.is_dir()):
            c = sub / f"{sub.name}.pdb"
            if c.exists():
                ligs.append(c)
    if not ligs:
        print("No ligands found.", file=sys.stderr)
        sys.exit(1)

    rows = []
    print(f"  {'ligand':<24s}{'atoms':>6s}{'required':>10s}{'fits box':>10s}"
          f"{'fits pocket':>13s}")
    n_tight = 0
    for lp in ligs:
        elems, xyz = read_heavy(lp)
        req = required_volume(elems, xyz, args.packing, args.samples)
        fits_free   = req <= box["free_volume"]
        fits_pocket = req <= box["pocket_volume"]
        if not fits_pocket:
            n_tight += 1
        rows.append({"ligand": lp.stem, "atoms": len(elems),
                     "required_volume": round(req, 1),
                     "fits_free": bool(fits_free),
                     "fits_pocket": bool(fits_pocket),
                     "pocket_margin": round(box["pocket_volume"] / req, 2) if req else None})
        print(f"  {lp.stem:<24s}{len(elems):>6d}{req:>9.0f}A"
              f"{('yes' if fits_free else 'NO'):>10s}"
              f"{('yes' if fits_pocket else 'NO'):>13s}")

    # Which cavities, not just the biggest, could hold each ligand?
    smallest_req = min((r["required_volume"] for r in rows), default=0.0)
    largest_req  = max((r["required_volume"] for r in rows), default=0.0)
    viable = [q for q in box["pockets"] if q["volume"] >= smallest_req]
    if box["pockets"]:
        print()
        print(f"  Cavities (top 5 of {box['n_pockets']}):")
        for q in box["pockets"][:5]:
            holds = sum(1 for r in rows if q["volume"] >= r["required_volume"])
            print(f"    {q['volume']:9.0f} A^3 at "
                  f"({q['centroid'][0]:7.2f},{q['centroid'][1]:7.2f},"
                  f"{q['centroid'][2]:7.2f})   holds {holds}/{len(rows)} ligand(s)")
        if len(viable) > 1:
            print(f"  {len(viable)} cavities can hold at least one ligand — "
                  f"the box may span more than one site.")

    print()
    if n_tight:
        print(f"  {n_tight} ligand(s) cannot fit the largest contiguous pocket.")
        print(f"  Docking will still return poses for them, typically draped")
        print(f"  across the receptor surface with plausible scores.")
        if viable:
            best = viable[0]
            print(f"  Largest cavity is at ({best['centroid'][0]:.2f}, "
                  f"{best['centroid'][1]:.2f}, {best['centroid'][2]:.2f}) — "
                  f"re-prep centred there if it is the intended site.")
        else:
            print(f"  No cavity in this box holds even the smallest ligand "
                  f"({smallest_req:.0f} A^3); the box is centred on protein "
                  f"interior or exterior rather than a site.")
    else:
        print(f"  All ligands fit the largest contiguous pocket.")

    if args.json:
        Path(args.json).write_text(json.dumps({"box": box, "ligands": rows}, indent=2))
        print(f"  JSON written: {args.json}")


if __name__ == "__main__":
    main()
