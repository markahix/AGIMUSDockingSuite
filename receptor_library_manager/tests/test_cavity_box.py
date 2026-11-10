#!/usr/bin/env python3
"""
Unit tests for cavity_box.py. Pure geometry plus a synthetic buried-cavity map
(no autogrid4, no external data). Run: python3 -I test_cavity_box.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cavity_box as cbx

FAIL = []


def check(c, m):
    print(f"  [{'PASS' if c else 'FAIL'}] {m}")
    if not c:
        FAIL.append(m)


def write_pdb(path, atoms):
    with open(path, "w") as f:
        for i, (rec, name, resname, chain, resseq, x, y, z) in enumerate(atoms, 1):
            f.write(f"{rec:<6}{i:>5} {name:<4} {resname:>3} {chain}{resseq:>4}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          C\n")
        f.write("END\n")


tmp = Path(tempfile.mkdtemp())

print("resolve_seed:")
# explicit wins
seed, src = cbx.resolve_seed(tmp / "none.pdb", tmp / "none.pdb",
                             {"center_x": 1, "center_y": 2, "center_z": 3})
check(src == "explicit" and np.allclose(seed, [1, 2, 3]), "explicit XYZ seed")

# native ligand centroid (HETATM LIG), skipping water
orig = tmp / "orig.pdb"
write_pdb(orig, [
    ("ATOM",   "CA",  "ALA", "A", 1,  0, 0, 0),
    ("HETATM", "O",   "HOH", "A", 50, 9, 9, 9),       # solvent, skipped
    ("HETATM", "C1",  "LIG", "A", 99, 2, 0, 0),
    ("HETATM", "C2",  "LIG", "A", 99, 4, 0, 0),       # LIG centroid = (3,0,0)
])
seed, src = cbx.resolve_seed(orig, orig, {})
check(src == "native-ligand" and np.allclose(seed, [3, 0, 0]),
      f"native ligand centroid, water skipped (got {seed}, {src})")

# residue-mask centroid on the fixed structure
fixed = tmp / "fixed.pdb"
write_pdb(fixed, [
    ("ATOM", "CA", "GLY", "A", 10, 0, 0, 0),
    ("ATOM", "CA", "SER", "A", 20, 6, 0, 0),          # masked residues 10,20
    ("ATOM", "CA", "LEU", "A", 30, 100, 0, 0),        # not masked
])
seed, src = cbx.resolve_seed(orig, fixed, {"site_residues": ["A:10", "A:20"]})
check(src == "residue-mask" and np.allclose(seed, [3, 0, 0]),
      f"residue-mask centroid (got {seed}, {src})")


print("box_from_points:")
pts = np.array([[-5, -1, -2], [5, 1, 2]], float)
b = cbx.box_from_points(pts, pad=3.0)
check(abs(b["cx"]) < 1e-6 and abs(b["sx"] - 16) < 1e-6,
      f"box centre/size with padding (sx={b['sx']})")


print("reorient_decision:")
# an elongated rod along the (1,1,1) diagonal: reorienting shrinks the box a lot
t = np.linspace(-15, 15, 80)
rod = np.outer(t, np.array([1, 1, 1]) / np.sqrt(3))
rod += np.random.default_rng(0).normal(scale=0.2, size=rod.shape)
R, T, oriented, gain = cbx.reorient_decision(rod, pad=4.0, gain_threshold=2.0)
check(R is not None and gain >= 2.0, f"elongated rod is reoriented (gain {gain})")
check(np.allclose(cbx.apply_transform(rod, T), oriented, atol=1e-6),
      "4x4 transform reproduces the oriented points")

# a near-cubic cloud: not worth reorienting
cube = np.random.default_rng(1).uniform(-8, 8, size=(400, 3))
R2, T2, oriented2, gain2 = cbx.reorient_decision(cube, pad=4.0, gain_threshold=2.0)
check(R2 is None and np.allclose(T2, np.eye(4)),
      f"cubic cloud keeps the original frame (gain {gain2})")


print("rasterize_core / core_contains:")
box = {"cx": 0, "cy": 0, "cz": 0, "sx": 10, "sy": 10, "sz": 10}
core_pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, -1, 0]], float)
mask = cbx.rasterize_core(core_pts, spacing=0.5, box=box)
check(cbx.core_contains([0.1, 0.1, 0.1], mask), "point in core is inside")
check(not cbx.core_contains([4.9, 4.9, 4.9], mask), "far corner is outside core")
check(not cbx.core_contains([100, 0, 0], mask), "point outside box is outside")


print("measure_from_map (synthetic buried cavity):")
# 40^3 grid at 0.5 A: protein everywhere, a free 6^3 cavity in the centre.
n = 40; sp = 0.5
arr = np.full((n, n, n), 10.0, dtype=np.float32)   # protein (energy >= threshold)
arr[17:23, 17:23, 17:23] = -1.0                    # buried free cavity
center = (0.0, 0.0, 0.0)
res = cbx.measure_from_map(arr, sp, center, (n, n, n),
                           np.array([0.0, 0.0, 0.0]), apron_reach=0.0,
                           reorient_gain=2.0)
b = res["box"]
check(2.0 < b["sx"] < 12.0 and 2.0 < b["sy"] < 12.0,
      f"box sized to the buried cavity, not the whole grid (size {b['sx']})")
check(res["core_volume"] > 0, f"core volume measured ({res['core_volume']})")
check(cbx.core_contains([0, 0, 0], res["core_mask"]),
      "cavity centre is inside the measured core")

print("measure_box (full chain via a mock autogrid):")


def _write_map(path, grid, spacing, cen):
    nz, ny, nx = grid.shape
    with open(path, "w") as f:
        f.write("GRID_PARAMETER_FILE receptor.gpf\n")
        f.write("GRID_DATA_FILE receptor.maps.fld\n")
        f.write("MACROMOLECULE receptor.pdbqt\n")
        f.write(f"SPACING {spacing:.3f}\n")
        f.write(f"NELEMENTS {nx-1} {ny-1} {nz-1}\n")
        f.write(f"CENTER {cen[0]:.3f} {cen[1]:.3f} {cen[2]:.3f}\n")
        for v in grid.reshape(-1):
            f.write(f"{float(v):.4f}\n")


def mock_gen(rigid, maps_dir, box, spacing, probes):
    maps_dir = Path(maps_dir)
    maps_dir.mkdir(parents=True, exist_ok=True)
    _write_map(maps_dir / f"receptor.{probes[0]}.map", arr, sp, center)
    return maps_dir / "receptor.maps.fld"


res2 = cbx.measure_box(tmp / "dummy.pdbqt", np.array([0.0, 0, 0]), sp, mock_gen,
                       tmp / "prov_work", provisional_size=20.0, apron_reach=0.0)
check(2.0 < res2["box"]["sx"] < 12.0,
      f"measure_box sized box to the cavity ({res2['box']['sx']})")
check(cbx.core_contains([0, 0, 0], res2["core_mask"]),
      "measure_box core contains the seed")


print("native-ligand validation:")
nat = cbx.native_ligand_atoms(orig)                     # LIG C1(2,0,0),C2(4,0,0)
check(len(nat) == 2, f"native ligand atoms found ({len(nat)})")
box_n = {"cx": 3, "cy": 0, "cz": 0, "sx": 8, "sy": 8, "sz": 8}
core_n = cbx.rasterize_core(nat, 0.5, box_n)
check(abs(cbx.fraction_in_core(nat, core_n) - 1.0) < 1e-9,
      "native ligand fully inside its own core")
check(cbx.fraction_in_core(np.array([[50.0, 50, 50]]), core_n) == 0.0,
      "far point not in core")
check(cbx.fraction_in_core(np.zeros((0, 3)), core_n) == 0.0,
      "empty point set is 0% in core")

print()
if FAIL:
    print(f"{len(FAIL)} FAILED")
    sys.exit(1)
print("all tests passed")
