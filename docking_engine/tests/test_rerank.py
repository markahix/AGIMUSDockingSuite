#!/usr/bin/env python3
"""
Test that dock_batch.build_complex re-ranks to the best physically VALID pose
instead of the lowest-energy one. Synthetic DLG + receptor, no external data.
Run: python3 -I test_rerank.py
"""
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dock_batch as db

FAIL = []


def check(c, m):
    print(f"  [{'PASS' if c else 'FAIL'}] {m}")
    if not c:
        FAIL.append(m)


def atom(serial, name, resname, resseq, x, y, z, elem, rec="ATOM"):
    # Standard PDB columns: name 13-16, altLoc 17, resName 18-20, chain 22,
    # resSeq 23-26, coords 31-54, element 77-78.
    return (f"{rec:<6}{serial:>5} {name:<4} {resname:>3} A{resseq:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2}\n")


def benzene(cx, cy, cz, r):
    return [(cx + r * math.cos(2 * math.pi * k / 6),
             cy + r * math.sin(2 * math.pi * k / 6), cz) for k in range(6)]


tmp = Path(tempfile.mkdtemp())

# Receptor: one PHE ring (large enough that a threading bond does not clash it)
names = ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]
ring = benzene(10, 10, 10, 2.6)
fixed = tmp / "fixed.pdb"
with open(fixed, "w") as f:
    for i, ((x, y, z), nm) in enumerate(zip(ring, names), 1):
        f.write(atom(i, nm, "PHE", 42, x, y, z, "C"))
    f.write("END\n")

# DLG with two runs:
#   run 1 (E = -10, best score): ligand bond threaded through the PHE ring
#   run 2 (E =  -8, worse):      ligand well away from the receptor (valid)
pierce = [("C1", 10.0, 10.0, 9.25), ("C2", 10.0, 10.0, 10.75)]
clean  = [("C1", 0.0, 0.0, 0.0),    ("C2", 1.5, 0.0, 0.0)]


def dlg_run(model, energy, lig):
    out = [f"DOCKED: MODEL {model}\n",
           f"DOCKED: USER    Estimated Free Energy of Binding    = "
           f"{energy:.2f} kcal/mol\n"]
    for i, (nm, x, y, z) in enumerate(lig, 1):
        out.append("DOCKED: " + atom(i, nm, "LIG", 1, x, y, z, "C", "ATOM"))
    out.append("DOCKED: ENDMDL\n")
    return out


dlg = tmp / "lig.dlg"
with open(dlg, "w") as f:
    f.writelines(dlg_run(1, -10.0, pierce))     # best energy, INVALID
    f.writelines(dlg_run(2, -8.0, clean))       # worse energy, valid

out_pdb = tmp / "complex_LIG.pdb"
res = db.build_complex(fixed, dlg, out_pdb, "LIG", n_lig_atoms=2,
                       flex_pdbqt=None, h_template=None)

print("build_complex re-rank:")
check(res["ok"], "build succeeded")
check(res["n_runs"] == 2, f"two runs parsed (got {res['n_runs']})")
check(res["reranked"], "reranked away from the lowest-energy pose")
check(res["rank"] == 1, f"chose the 2nd-best pose (rank {res['rank']})")
check(abs(res["energy"] - (-8.0)) < 1e-6,
      f"reported energy is the kept pose's -8.0 (got {res['energy']})")
check(res["status"].startswith("OK"), f"status OK ({res['status']})")

# Control: if the only run is the pierced one, it must fail (no valid fallback)
dlg2 = tmp / "lig2.dlg"
with open(dlg2, "w") as f:
    f.writelines(dlg_run(1, -10.0, pierce))
res2 = db.build_complex(fixed, dlg2, tmp / "c2.pdb", "LIG", n_lig_atoms=2)
print("single invalid pose:")
check(not res2["reranked"] and "PIERCE" in res2["status"],
      f"lone pierced pose is reported, not hidden ({res2['status']})")

print()
if FAIL:
    print(f"{len(FAIL)} FAILED")
    sys.exit(1)
print("all tests passed")
