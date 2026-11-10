#!/usr/bin/env python3
"""
Unit tests for complex_validity.py. Run: python3 -I test_complex_validity.py
Synthetic geometry only - no external data, no network.
"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import complex_validity as cv

FAIL = []


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAIL.append(msg)


def benzene(center=(0, 0, 0), r=1.39, z_axis=(0, 0, 1)):
    """Six carbons of a planar ring, centred at `center`, normal along z_axis."""
    cx, cy, cz = center
    pts = []
    for k in range(6):
        a = 2 * math.pi * k / 6
        pts.append([cx + r * math.cos(a), cy + r * math.sin(a), cz])
    return np.array(pts)


print("segment_pierces_ring:")
C, n, rad = cv.ring_plane(benzene())
check(abs(abs(n[2]) - 1.0) < 1e-6, f"benzene normal is z ({n.round(3).tolist()})")
check(abs(rad - 1.39) < 0.05, f"benzene radius ~1.39 (got {rad:.3f})")

p_through, d1 = cv.segment_pierces_ring(np.array([0, 0, -3.0]),
                                        np.array([0, 0, 3.0]), C, n, rad)
check(p_through and d1 < 0.01, "bond along axis through centre pierces")

p_miss, d2 = cv.segment_pierces_ring(np.array([5, 0, -3.0]),
                                     np.array([5, 0, 3.0]), C, n, rad)
check(not p_miss, f"bond crossing 5 A off-centre does not pierce (d={d2})")

p_par, _ = cv.segment_pierces_ring(np.array([0, 0, 2.0]),
                                   np.array([3, 0, 2.0]), C, n, rad)
check(not p_par, "bond parallel to the ring plane does not pierce")

p_short, _ = cv.segment_pierces_ring(np.array([0, 0, 1.0]),
                                     np.array([0, 0, 3.0]), C, n, rad)
check(not p_short, "bond that does not reach the plane does not pierce")


print("bond/ring perception:")
# benzene as atom records, consecutive carbons bonded by proximity
atoms = [{"xyz": p, "ad4": "A", "elem": "C", "name": f"C{i}",
          "resname": "LIG", "resid": "1", "is_h": False}
         for i, p in enumerate(benzene())]
bonds, heavy = cv.infer_bonds(atoms)
rings = cv.find_rings(bonds, len(heavy), 8)
check(len(rings) == 1 and len(rings[0]) == 6, f"one 6-ring perceived "
      f"({len(rings)} ring(s), sizes {[len(r) for r in rings]})")


print("clash counting:")
lig = [{"xyz": np.array([0.0, 0, 0]), "ad4": "C", "elem": "C", "is_h": False}]
rec_far = [{"xyz": np.array([5.0, 0, 0]), "ad4": "C", "elem": "C", "is_h": False}]
rec_near = [{"xyz": np.array([1.0, 0, 0]), "ad4": "C", "elem": "C", "is_h": False}]
check(cv.count_clashes(lig, rec_far) == 0, "atoms 5 A apart: no clash")
check(cv.count_clashes(lig, rec_near) == 1, "atoms 1 A apart: one clash")


print("piercing is caught WITHOUT relying on clash:")
# Receptor PHE ring in the z=10 plane around (10,10,10); large so the threading
# bond's endpoints sit >2.5 A from every ring atom (no clash), yet the bond
# still crosses the ring disk.
big = benzene(center=(10, 10, 10), r=2.6)
phe = [{"xyz": p, "ad4": "A", "elem": "C", "name": nm,
        "resname": "PHE", "resid": "42", "is_h": False}
       for p, nm in zip(big, ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"))]
# ligand bond straight through the centre along z, endpoints 1.5 A apart
ligand = [
    {"xyz": np.array([10.0, 10.0, 9.25]), "ad4": "C", "elem": "C",
     "name": "C1", "resname": "LIG", "resid": "1", "is_h": False},
    {"xyz": np.array([10.0, 10.0, 10.75]), "ad4": "C", "elem": "C",
     "name": "C2", "resname": "LIG", "resid": "1", "is_h": False},
]
clashes = cv.count_clashes(ligand, phe)
pierces = cv.find_piercings(ligand, phe)
check(clashes == 0, f"threading bond does not clash any ring atom "
      f"(clashes={clashes})")
check(len(pierces) == 1 and pierces[0]["ring"].startswith("PHE"),
      f"threading bond is flagged as piercing PHE ({pierces})")
v = cv.validate(ligand, phe)
check(not v["valid"], "verdict INVALID on a clash-free ring-pierce")


print("clean pose is valid:")
clean = [
    {"xyz": np.array([10.0, 10.0, 15.0]), "ad4": "C", "elem": "C",
     "name": "C1", "resname": "LIG", "resid": "1", "is_h": False},
    {"xyz": np.array([11.5, 10.0, 15.0]), "ad4": "C", "elem": "C",
     "name": "C2", "resname": "LIG", "resid": "1", "is_h": False},
]
v = cv.validate(clean, phe)
check(v["valid"], f"well-separated ligand is valid ({v})")


print("core-occupancy filter:")
# one-cell core at the origin: cell (0,0,0) of a 2^3 grid, spacing 1.0
core = {"shape": [2, 2, 2], "spacing": 1.0, "origin": [-1, -1, -1],
        "runs": [[0, 1]]}
atom_in = [{"xyz": np.array([-0.5, -0.5, -0.5]), "is_h": False,
            "ad4": "C", "elem": "C", "name": "C1"}]
atom_out = [{"xyz": np.array([0.5, 0.5, 0.5]), "is_h": False,
             "ad4": "C", "elem": "C", "name": "C1"}]
check(cv.core_occupancy(atom_in, core) == 1.0, "atom in core = 100%")
check(cv.core_occupancy(atom_out, core) == 0.0, "atom outside core = 0%")
check(cv.core_occupancy(atom_in, None) == 1.0, "no mask = no constraint")
# a clash-free, pierce-free pose parked outside the core is still INVALID
vout = cv.validate(atom_out, [], core_mask=core, min_core=0.3)
check(not vout["valid"] and not vout["core_ok"],
      f"surface-parked clean pose is rejected ({vout['core_fraction']})")
vin = cv.validate(atom_in, [], core_mask=core, min_core=0.3)
check(vin["valid"], "in-core clean pose is valid")
check(cv.validate(atom_out, [])["valid"],
      "without a mask the same pose passes (backward compatible)")


print()
if FAIL:
    print(f"{len(FAIL)} FAILED")
    sys.exit(1)
print("all tests passed")
