#!/usr/bin/env python3
"""
torsion_audit.py — measure the effective torsional dimensionality of a ligand.

A ligand's nominal rotatable-bond count overstates the search space it
actually presents. Three effects reduce it, each detectable from the
molecular graph and geometry alone, with no chemical-class lookup:

  symmetry    Rotating a branch onto itself makes the torsion periodic at
              360/n. An SO3, CF3, t-butyl, nitro, carboxylate or phenyl
              branch needs only 360/n degrees of scan, exactly.
  reach       A torsion whose branch sweeps a small radius barely moves any
              heavy atom, so a few states cover its accessible positions.
  barrier     Amide, ester, conjugated and anomeric bonds are not freely
              rotating; they occupy a small number of minima.

Every collapse is reported with the maximum heavy-atom displacement it
discards, so the reduction is measured per molecule rather than assumed.
Nothing here is specific to a chemical class: predicates are computed from
connectivity and coordinates, so the same analysis applies to any ligand.

Usage:
    python3 torsion_audit.py --pdb LIGAND.pdb [options]

Options:
    --pdb PATH            Ligand PDB to analyse (required)
    --resolution DEG      Angular resolution a free torsion would need
                          (default: 30)
    --reach-cutoff AA     Swept radius below which a torsion is treated as
                          negligible (default: 1.5)
    --tolerance AA        Max heavy-atom displacement a collapse may discard
                          (default: 0.25)
    --json PATH           Write the full audit as JSON
    --quiet               Suppress the table, print the summary line only
    --help                Show this message
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Covalent radii (Angstrom) for bond perception.
_COV = {"C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07,
        "F": 0.57, "CL": 1.02, "BR": 1.20, "I": 1.39, "H": 0.31}
_BOND_TOL = 0.45

# Symmetry orders worth testing, highest first.
_SYM_ORDERS = (6, 4, 3, 2)
_SYM_TOL    = 0.30          # Angstrom, positional match after rotation


# ══════════════════════════════════════════════════════════════════════════════
# Structure parsing and bond perception
# ══════════════════════════════════════════════════════════════════════════════

def read_heavy_atoms(pdb_path):
    """
    Read heavy atoms from a PDB.

    Returns (names, elements, coords) where coords is an (N,3) array.
    Hydrogens are excluded: torsions that move only hydrogens cannot change
    a heavy-atom score and are not part of the search space considered here.
    """
    names, elems, xyz = [], [], []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            name = line[12:16].strip()
            el   = (line[76:78].strip() or "").upper()
            if not el:
                raw = "".join(c for c in name if c.isalpha())
                el  = raw[:1].upper()
            if el in ("H", "D"):
                continue
            try:
                xyz.append([float(line[30:38]), float(line[38:46]),
                            float(line[46:54])])
            except ValueError:
                continue
            names.append(name)
            elems.append(el)
    return names, elems, np.asarray(xyz, dtype=float)


def perceive_bonds(elems, xyz):
    """Distance-based bond perception. Returns adjacency as list of sets."""
    n   = len(elems)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cutoff = (_COV.get(elems[i], 0.76) + _COV.get(elems[j], 0.76)
                      + _BOND_TOL)
            if float(np.linalg.norm(xyz[i] - xyz[j])) < cutoff:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def find_ring_bonds(adj, n):
    """Bonds that lie in a ring: an alternative path exists between the ends."""
    ring = set()
    for u in range(n):
        for v in adj[u]:
            if v < u:
                continue
            seen, frontier = {u}, [u]
            found = False
            for _ in range(10):
                nxt = []
                for x in frontier:
                    for y in adj[x]:
                        if x == u and y == v:
                            continue
                        if y == v:
                            found = True
                            break
                        if y not in seen:
                            seen.add(y)
                            nxt.append(y)
                    if found:
                        break
                if found or not nxt:
                    break
                frontier = nxt
            if found:
                ring.add((u, v))
    return ring


def ring_systems(adj, ring_bonds, n):
    """Connected components of ring atoms; each is one rigid fragment."""
    ring_atoms = set()
    for u, v in ring_bonds:
        ring_atoms.add(u)
        ring_atoms.add(v)
    comp, cid = {}, 0
    for a in sorted(ring_atoms):
        if a in comp:
            continue
        stack, comp[a] = [a], cid
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y in ring_atoms and y not in comp:
                    comp[y] = cid
                    stack.append(y)
        cid += 1
    return comp, cid


def rotatable_bonds(adj, ring_bonds, n):
    """
    Single, acyclic bonds with a non-terminal atom at each end.
    Terminal branches move no heavy atom relative to the bond axis in a way
    that matters, and ring bonds are handled by ring-pucker analysis instead.
    """
    out = []
    for u in range(n):
        for v in adj[u]:
            if v < u or (u, v) in ring_bonds:
                continue
            if len(adj[u]) > 1 and len(adj[v]) > 1:
                out.append((u, v))
    return out


def split_branch(adj, u, v, n):
    """Atoms reachable from v without crossing the u-v bond."""
    seen, stack = {v}, [v]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if x == v and y == u:
                continue
            if y not in seen:
                seen.add(y)
                stack.append(y)
    seen.discard(v)
    return seen | {v}


# ══════════════════════════════════════════════════════════════════════════════
# Per-torsion predicates
# ══════════════════════════════════════════════════════════════════════════════

def _rotate_about(points, origin, axis, angle):
    axis = axis / np.linalg.norm(axis)
    p    = points - origin
    return (origin + p * math.cos(angle)
            + np.cross(axis, p) * math.sin(angle)
            + np.outer(p @ axis, axis) * (1.0 - math.cos(angle)))


def symmetry_order(branch_idx, elems, xyz, origin, axis):
    """
    Largest n for which rotating the branch by 360/n maps its heavy atoms
    onto heavy atoms of the same element. The torsion is then periodic at
    360/n and only that much of it needs scanning. Exact, not heuristic.
    """
    if len(branch_idx) < 2:
        return 1
    pts  = xyz[branch_idx]
    els  = [elems[i] for i in branch_idx]
    for n in _SYM_ORDERS:
        rot = _rotate_about(pts, origin, axis, 2.0 * math.pi / n)
        ok  = True
        for k, p in enumerate(rot):
            d = np.linalg.norm(pts - p, axis=1)
            d[[m for m, e in enumerate(els) if e != els[k]]] = 9e9
            if d.min() > _SYM_TOL:
                ok = False
                break
        if ok:
            return n
    return 1


def swept_radius(branch_idx, xyz, origin, axis):
    """Max perpendicular distance from the bond axis to any branch atom."""
    if not branch_idx:
        return 0.0
    axis = axis / np.linalg.norm(axis)
    rel  = xyz[branch_idx] - origin
    perp = rel - np.outer(rel @ axis, axis)
    return float(np.linalg.norm(perp, axis=1).max())


def barrier_type(u, v, adj, elems, xyz):
    """
    Detect bonds that are not freely rotating, from local connectivity.
    Returns (label, n_states) or (None, 0) if the bond is free.
    """
    def double_bonded_O(c):
        for nb in adj[c]:
            if elems[nb] == "O" and len(adj[nb]) == 1:
                if float(np.linalg.norm(xyz[c] - xyz[nb])) < 1.30:
                    return True
        return False

    eu, ev = elems[u], elems[v]

    # Amide / ester: C(=O)-N or C(=O)-O. Planar, effectively trans.
    for a, b in ((u, v), (v, u)):
        if elems[a] == "C" and elems[b] in ("N", "O") and double_bonded_O(a):
            return ("amide/ester", 2)

    # Anomeric / acetal: C bearing two or more oxygens, rotating about C-O.
    for a, b in ((u, v), (v, u)):
        if elems[a] == "C" and elems[b] == "O":
            if sum(1 for nb in adj[a] if elems[nb] == "O") >= 2:
                return ("anomeric", 3)

    # Conjugated: both ends sp2-like (three heavy neighbours, near planar).
    def sp2(c):
        nbs = [x for x in adj[c] if x != (v if c == u else u)]
        if len(nbs) < 2:
            return False
        v1 = xyz[nbs[0]] - xyz[c]
        v2 = xyz[nbs[1]] - xyz[c]
        nrm = np.cross(v1, v2)
        if np.linalg.norm(nrm) < 1e-6:
            return True
        other = xyz[(v if c == u else u)] - xyz[c]
        nrm = nrm / np.linalg.norm(nrm)
        return abs(float(other @ nrm) / max(np.linalg.norm(other), 1e-9)) < 0.35

    if eu == "C" and ev == "C" and sp2(u) and sp2(v):
        return ("conjugated", 2)

    return (None, 0)


def correlated_pairs(rot_bonds, adj, elems, comp):
    """
    Two torsions about a divalent sp3 bridge (-O-, -S-, -NH-) joining two
    rigid fragments are sterically coupled; their joint effect is a 2D map,
    not two independent scans. This is the general form of the C-O-C case.
    """
    by_atom = defaultdict(list)
    for (u, v) in rot_bonds:
        by_atom[u].append((u, v))
        by_atom[v].append((u, v))
    pairs = []
    for a, bonds in by_atom.items():
        if elems[a] not in ("O", "S", "N"):
            continue
        if len(adj[a]) != 2 or len(bonds) != 2:
            continue
        ends = [v if u == a else u for (u, v) in bonds]
        if all(e in comp for e in ends):          # both sides ring fragments
            pairs.append((tuple(bonds[0]), tuple(bonds[1]), a))
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# Audit
# ══════════════════════════════════════════════════════════════════════════════

def audit_ligand(pdb_path, resolution=30.0, reach_cutoff=1.5,
                 tolerance=0.25, use_barriers=False):
    names, elems, xyz = read_heavy_atoms(pdb_path)
    n = len(elems)
    if n == 0:
        raise ValueError(f"no heavy atoms read from {pdb_path}")

    adj        = perceive_bonds(elems, xyz)
    ring_bonds = find_ring_bonds(adj, n)
    comp, nring = ring_systems(adj, ring_bonds, n)
    rots       = rotatable_bonds(adj, ring_bonds, n)
    pairs      = correlated_pairs(rots, adj, elems, comp)
    paired     = {b for p in pairs for b in p[:2]}

    free_states = max(1, int(round(360.0 / resolution)))
    entries     = []

    for (u, v) in rots:
        # Rotate the smaller side.
        side_v = split_branch(adj, u, v, n)
        side_u = split_branch(adj, v, u, n)
        branch = sorted(side_v if len(side_v) <= len(side_u) else side_u)
        pivot, other = (u, v) if branch is not None and v in branch else (v, u)
        origin = xyz[pivot]
        axis   = xyz[other] - xyz[pivot]
        moving = [i for i in branch if i != pivot]

        sym   = symmetry_order(moving, elems, xyz, origin, axis) if moving else 1
        reach = swept_radius(moving, xyz, origin, axis)
        btype, bstates = barrier_type(u, v, adj, elems, xyz)

        # Symmetry first: it reduces the PERIOD and is exact, because the
        # rotated heavy atoms coincide with the originals. Tolerance then
        # sets the sampling density within that period. Treating symmetry
        # as just another approximation understates it and mis-orders the
        # two effects.
        period = 360.0 / max(1, sym)

        # States required so the worst angular error, period/(2k), displaces
        # no atom further than the tolerance:
        #     2 R sin(period / 4k) <= tol
        def states_for(radius, per_deg):
            if radius <= 1e-6:
                return 1
            ratio = tolerance / (2.0 * radius)
            if ratio >= 1.0:
                return 1
            k = math.radians(per_deg) / (4.0 * math.asin(ratio))
            return max(1, int(math.ceil(k)))

        cap = max(1, int(round(period / resolution)))   # density ceiling

        if not moving:
            states, kind, reason, disc = 1, "exact", "no heavy atoms move", 0.0
        elif use_barriers and btype:
            # A chemical restriction, not a numerical approximation: the
            # intermediate angles are asserted to be unpopulated. Off by
            # default so that correctness never depends on the predicate.
            states, kind = bstates, "physical"
            reason, disc = btype, 0.0
        else:
            needed = states_for(reach, period)
            states = min(cap, needed)
            kind   = "exact" if (sym > 1 and states >= needed) else "approx"
            disc   = (0.0 if states >= needed
                      else 2.0 * reach * math.sin(math.radians(period / (4.0 * states))))
            bits = []
            if sym > 1:
                bits.append(f"{sym}-fold symmetric")
            bits.append(f"reach {reach:.2f} A")
            if btype:
                bits.append(f"[{btype} detected, not applied]")
            reason = ", ".join(bits)

        entries.append({
            "bond":        [names[u], names[v]],
            "elements":    [elems[u], elems[v]],
            "branch_size": len(moving),
            "symmetry":    sym,
            "reach":       round(reach, 3),
            "barrier":     btype,
            "states":      states,
            "reason":      reason,
            "kind":        kind,
            "discarded":   round(disc, 3),
            "correlated":  ((u, v) in paired or (v, u) in paired),
            "exceeds_tol": bool(disc > tolerance),
        })

    nominal   = len(rots)
    naive     = free_states ** nominal if nominal else 1
    reduced   = 1
    for e in entries:
        reduced *= max(1, e["states"])
    eff_dof = (math.log(reduced) / math.log(free_states)) if reduced > 1 else 0.0

    return {
        "ligand":          Path(pdb_path).stem,
        "heavy_atoms":     n,
        "ring_systems":    nring,
        "resolution_deg":  resolution,
        "nominal_torsions": nominal,
        "effective_dof":   round(eff_dof, 2),
        "naive_states":    naive,
        "reduced_states":  reduced,
        "reduction_factor": (naive / reduced) if reduced else 1,
        "correlated_pairs": [
            {"bridge": names[p[2]],
             "bonds": [[names[p[0][0]], names[p[0][1]]],
                       [names[p[1][0]], names[p[1][1]]]]}
            for p in pairs],
        "torsions":        entries,
        "max_discarded":   round(max([e["discarded"] for e in entries], default=0.0), 3),
        "over_tolerance":  sum(1 for e in entries if e["exceeds_tol"]),
    }


def torsion_period_map(audit):
    """
    {(nameA,nameB): period_degrees} — the unique angular range per torsion.
    Used to fold conformer torsion angles into their symmetry-reduced form
    so that effectively-identical conformers compare equal.
    """
    out = {}
    for e in audit["torsions"]:
        out[(e["bond"][0], e["bond"][1])] = 360.0 / max(1, e["symmetry"])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

def format_report(a):
    L = []
    L.append(f"  Ligand        : {a['ligand']}")
    L.append(f"  Heavy atoms   : {a['heavy_atoms']}   ring systems: {a['ring_systems']}")
    L.append(f"  Torsions      : {a['nominal_torsions']} nominal"
             f"  ->  {a['effective_dof']} effective DOF"
             f"  (at {a['resolution_deg']:.0f} deg)")
    if a["reduced_states"] and a["reduction_factor"] > 1:
        L.append(f"  Search space  : {a['naive_states']:.3g}"
                 f"  ->  {a['reduced_states']:.3g}"
                 f"   ({a['reduction_factor']:.3g}x smaller)")
    L.append(f"  Max discarded : {a['max_discarded']:.3f} A"
             f"   ({a['over_tolerance']} torsion(s) over tolerance)")
    if a["correlated_pairs"]:
        L.append(f"  Correlated    : {len(a['correlated_pairs'])} divalent-bridge pair(s)"
                 f" — sample jointly, not independently")
        for p in a["correlated_pairs"]:
            L.append(f"      bridge {p['bridge']}: "
                     f"{'-'.join(p['bonds'][0])} + {'-'.join(p['bonds'][1])}")
    L.append("")
    L.append(f"  {'bond':<14s}{'branch':>7s}{'sym':>5s}{'reach':>7s}"
             f"{'states':>7s}{'disc.A':>8s}  reason")
    for e in sorted(a["torsions"], key=lambda x: -x["states"]):
        flag = " *" if e["exceeds_tol"] else ""
        corr = " [corr]" if e["correlated"] else ""
        L.append(f"  {'-'.join(e['bond']):<14s}{e['branch_size']:>7d}"
                 f"{e['symmetry']:>5d}{e['reach']:>7.2f}{e['states']:>7d}"
                 f"{e['discarded']:>8.2f}  {e['reason']}{corr}{flag}")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(
        prog="torsion_audit.py",
        description="Measure the effective torsional dimensionality of a ligand.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb", metavar="PATH",
                   help="Ligand PDB to analyse")
    p.add_argument("--library", metavar="DIR",
                   help="Audit every <DIR>/<name>/<name>.pdb in a library "
                        "directory and print a ranked summary")
    p.add_argument("--resolution", type=float, default=30.0, metavar="DEG",
                   help="Angular resolution a free torsion would need (default: 30)")
    p.add_argument("--reach-cutoff", type=float, default=1.5, metavar="AA",
                   help="Swept radius below which a torsion is negligible (default: 1.5)")
    p.add_argument("--tolerance", type=float, default=0.25, metavar="AA",
                   help="Max heavy-atom displacement a collapse may discard (default: 0.25)")
    p.add_argument("--use-barriers", action="store_true",
                   help="Also collapse amide/ester/anomeric/conjugated bonds to "
                        "their minima. Off by default: this is a chemical claim, "
                        "not a geometric one, so correctness must not depend on it")
    p.add_argument("--json", metavar="PATH", help="Write the full audit as JSON")
    p.add_argument("--quiet", action="store_true",
                   help="Print the summary line only")
    args = p.parse_args()

    if not args.pdb and not args.library:
        print("ERROR: give --pdb or --library (see --help)", file=sys.stderr)
        sys.exit(1)

    if args.library:
        lib = Path(args.library)
        if not lib.is_dir():
            print(f"ERROR: not a directory: {lib}", file=sys.stderr)
            sys.exit(1)
        rows = []
        for sub in sorted(x for x in lib.iterdir() if x.is_dir()):
            cand = sub / f"{sub.name}.pdb"
            if not cand.exists():
                continue
            try:
                rows.append(audit_ligand(cand, args.resolution,
                                         args.reach_cutoff, args.tolerance,
                                         args.use_barriers))
            except Exception as exc:
                print(f"  {sub.name}: FAILED ({exc})", file=sys.stderr)
        if not rows:
            print("No ligands found.", file=sys.stderr)
            sys.exit(1)
        rows.sort(key=lambda r: -r["effective_dof"])
        print(f"  {'ligand':<24s}{'heavy':>6s}{'rings':>6s}"
              f"{'nominal':>8s}{'effDOF':>8s}{'reduction':>11s}")
        for r in rows:
            print(f"  {r['ligand']:<24s}{r['heavy_atoms']:>6d}"
                  f"{r['ring_systems']:>6d}{r['nominal_torsions']:>8d}"
                  f"{r['effective_dof']:>8.2f}{r['reduction_factor']:>10.4g}x")
        tot_n = sum(r["nominal_torsions"] for r in rows)
        tot_e = sum(r["effective_dof"] for r in rows)
        print(f"\n  {len(rows)} ligand(s):  {tot_n} nominal torsions  ->  "
              f"{tot_e:.1f} effective  "
              f"({100*(1-tot_e/max(tot_n,1)):.0f}% reduction)")
        if args.json:
            Path(args.json).write_text(json.dumps(rows, indent=2))
            print(f"  JSON written: {args.json}")
        sys.exit(0)

    pdb = Path(args.pdb)
    if not pdb.exists():
        print(f"ERROR: not found: {pdb}", file=sys.stderr)
        sys.exit(1)

    a = audit_ligand(pdb, args.resolution, args.reach_cutoff,
                     args.tolerance, args.use_barriers)

    if args.quiet:
        print(f"{a['ligand']}: {a['nominal_torsions']} nominal -> "
              f"{a['effective_dof']} effective DOF, "
              f"{a['reduction_factor']:.3g}x smaller search space")
    else:
        print(format_report(a))

    if args.json:
        Path(args.json).write_text(json.dumps(a, indent=2))
        if not args.quiet:
            print(f"\n  JSON written: {args.json}")


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# Conformer filtering
# ══════════════════════════════════════════════════════════════════════════════

def measure_torsion(xyz, a, b, c, d):
    """Dihedral a-b-c-d in degrees."""
    b0 = xyz[a] - xyz[b]
    b1 = xyz[c] - xyz[b]
    b2 = xyz[d] - xyz[c]
    b1 = b1 / max(float(np.linalg.norm(b1)), 1e-9)
    v  = b0 - (b0 @ b1) * b1
    w  = b2 - (b2 @ b1) * b1
    return math.degrees(math.atan2(float(np.cross(b1, v) @ w), float(v @ w)))


def torsion_fingerprint(xyz, adj, rots, periods_by_index, bin_deg=30.0):
    """
    Symmetry-folded torsion fingerprint.

    Each torsion angle is reduced modulo its own period, so two conformers
    differing only by rotation of a symmetric group (an SO3 turned 120 deg,
    say) collapse to the same fingerprint. Comparing raw Cartesian RMSD
    would call those distinct; they are not, because the heavy atoms occupy
    identical positions.
    """
    fp = []
    for (u, v) in rots:
        nu = [x for x in adj[u] if x != v]
        nv = [x for x in adj[v] if x != u]
        if not nu or not nv:
            fp.append(0)
            continue
        ang    = measure_torsion(xyz, nu[0], u, v, nv[0])
        period = periods_by_index.get((u, v), 360.0)
        fp.append(int(round((ang % period) / bin_deg)) % max(1, int(round(period / bin_deg))))
    return tuple(fp)


def internal_clash_count(xyz, elems, adj, scale=0.8, min_sep=4):
    """
    Count non-bonded heavy-atom pairs closer than scale * (vdW_i + vdW_j).

    Pairs separated by fewer than min_sep bonds are skipped: their distance
    is fixed by the covalent geometry, not by the torsions, so flagging them
    would reject valid conformers for their own bond lengths.
    """
    VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
           "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98}
    n = len(elems)

    # bond-graph distance up to min_sep
    near = [set() for _ in range(n)]
    for i in range(n):
        frontier, seen = {i}, {i}
        for _ in range(min_sep):
            nxt = set()
            for x in frontier:
                nxt |= adj[x]
            nxt -= seen
            seen |= nxt
            frontier = nxt
        near[i] = seen

    clashes = 0
    for i in range(n):
        for j in range(i + 1, n):
            if j in near[i]:
                continue
            thr = scale * (VDW.get(elems[i], 1.7) + VDW.get(elems[j], 1.7))
            if float(np.linalg.norm(xyz[i] - xyz[j])) < thr:
                clashes += 1
    return clashes


def filter_conformers(conformer_paths, reference_pdb,
                      bin_deg=30.0, max_internal_clashes=0):
    """
    Reduce a conformer set by discarding self-clashing geometries and
    symmetry-equivalent duplicates.

    Returns (kept, dropped_clash, dropped_duplicate) as lists of paths.
    Ordering is preserved, so the first representative of each distinct
    fingerprint is the one retained.
    """
    names, elems, xyz = read_heavy_atoms(reference_pdb)
    adj        = perceive_bonds(elems, xyz)
    ring_bonds = find_ring_bonds(adj, len(elems))
    rots       = rotatable_bonds(adj, ring_bonds, len(elems))

    periods = {}
    for (u, v) in rots:
        side_v = split_branch(adj, u, v, len(elems))
        side_u = split_branch(adj, v, u, len(elems))
        branch = sorted(side_v if len(side_v) <= len(side_u) else side_u)
        pivot, other = (u, v) if v in branch else (v, u)
        moving = [i for i in branch if i != pivot]
        sym = symmetry_order(moving, elems, xyz,
                             xyz[pivot], xyz[other] - xyz[pivot]) if moving else 1
        periods[(u, v)] = 360.0 / max(1, sym)

    kept, clash, dup = [], [], []
    seen = set()
    for path in conformer_paths:
        _, ce, cx = read_heavy_atoms(path)
        if len(cx) != len(xyz):
            kept.append(path)                 # atom count mismatch: keep, flag upstream
            continue
        if internal_clash_count(cx, ce, adj) > max_internal_clashes:
            clash.append(path)
            continue
        fp = torsion_fingerprint(cx, adj, rots, periods, bin_deg)
        if fp in seen:
            dup.append(path)
            continue
        seen.add(fp)
        kept.append(path)
    return kept, clash, dup
