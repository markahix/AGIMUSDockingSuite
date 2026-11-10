#!/usr/bin/env python3
"""
prep_receptor.py — receptor preparation for receptor-library-manager.
Ported directly from the working monolith prep_receptor.py (CACHEBUST=60).
Adapted for split-container architecture with SQLite state management.

Usage:
    python3 prep_receptor.py \
        --pdb /receptors/inbox/receptor.pdb \
        --receptors_dir /receptors \
        --library_dir /library \
        --center_x 46.2 --center_y 37.8 --center_z 16.1 \
        --autobox --padding 10.0 \
        [--name heparanase]
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from receptor_identity import extract_sequence, match_receptor, suggest_name
from receptor_state import ReceptorState
import cavity_box

# ── Constants (verbatim from working monolith) ─────────────────────────────────
_BACKBONE = {"N", "CA", "C", "O", "OXT", "H", "HA", "1H", "2H", "3H"}

_SC_BONDS = {
    "ARG": [("CA","CB"),("CB","CG"),("CG","CD"),("CD","NE"),("NE","CZ")],
    "ASN": [("CA","CB"),("CB","CG")],
    "ASP": [("CA","CB"),("CB","CG")],
    "CYS": [("CA","CB")],
    "GLN": [("CA","CB"),("CB","CG"),("CG","CD")],
    "GLU": [("CA","CB"),("CB","CG"),("CG","CD")],
    "HIS": [("CA","CB"),("CB","CG")],
    "ILE": [("CA","CB"),("CB","CG1")],
    "LEU": [("CA","CB"),("CB","CG")],
    "LYS": [("CA","CB"),("CB","CG"),("CG","CD"),("CD","CE"),("CE","NZ")],
    "MET": [("CA","CB"),("CB","CG"),("CG","SD")],
    "PHE": [("CA","CB"),("CB","CG")],
    "SER": [("CA","CB")],
    "THR": [("CA","CB")],
    "TRP": [("CA","CB"),("CB","CG")],
    "TYR": [("CA","CB"),("CB","CG")],
    "VAL": [("CA","CB")],
}

_AD4_TYPE_PRIORITY = [
    "A","C","HD","NA","N","OA","SA","S","P","F","Cl","Br","I",
    "Zn","Fe","Mg","Ca","Mn",
]
_DEFAULT_AD4_TYPES = _AD4_TYPE_PRIORITY[:12]


# ══════════════════════════════════════════════════════════════════════════════
# PDB splitting (multi-model)
# ══════════════════════════════════════════════════════════════════════════════

def split_models(pdb_path: Path) -> list[tuple[int, list[str]]]:
    models, current_model, current_lines, has_models = [], None, [], False
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                has_models = True
                try:    current_model = int(line.split()[1])
                except: current_model = len(models) + 1
                current_lines = [line]
            elif line.startswith("ENDMDL"):
                current_lines.append(line)
                models.append((current_model, current_lines))
                current_model, current_lines = None, []
            elif current_model is not None:
                current_lines.append(line)
            elif not has_models:
                current_lines.append(line)
    if not has_models:
        return [(1, current_lines)]
    return models or [(1, current_lines)]


def strip_model_records(lines: list[str]) -> list[str]:
    return [l for l in lines if not l.startswith(("MODEL","ENDMDL"))]


# ══════════════════════════════════════════════════════════════════════════════
# Superposition (verbatim from working monolith)
# ══════════════════════════════════════════════════════════════════════════════

def get_active_site_residues(lines: list[str], box: dict,
                              padding: float = 2.0) -> set[tuple]:
    cx, cy, cz = box["cx"], box["cy"], box["cz"]
    sx, sy, sz = box["sx"], box["sy"], box["sz"]
    residues = set()
    for line in lines:
        if not line.startswith("ATOM"): continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            if (abs(x-cx) <= sx/2+padding and
                abs(y-cy) <= sy/2+padding and
                abs(z-cz) <= sz/2+padding):
                residues.add((line[21].strip(), line[22:26].strip()))
        except (ValueError, IndexError):
            continue
    return residues


def superpose_to_reference(ref_lines: list[str], mob_lines: list[str],
                            model_num: int,
                            site_residues: set[tuple] = None) -> tuple[list[str], float, float]:
    def parse_ca(lines, restrict=None):
        ca = {}
        for line in lines:
            if not line.startswith("ATOM"): continue
            if line[12:16].strip() != "CA": continue
            try:
                chain  = line[21].strip()
                resseq = line[22:26].strip()
                key    = (chain, resseq)
                if restrict and key not in restrict: continue
                ca[key] = np.array([float(line[30:38]),
                                     float(line[38:46]),
                                     float(line[46:54])])
            except (ValueError, IndexError):
                continue
        return ca

    ref_ca = parse_ca(ref_lines, site_residues)
    mob_ca = parse_ca(mob_lines, site_residues)
    common = sorted(set(ref_ca.keys()) & set(mob_ca.keys()))
    if len(common) < 3:
        return mob_lines, 0.0, 0.0

    ref_pts = np.array([ref_ca[k] for k in common])
    mob_pts = np.array([mob_ca[k] for k in common])
    rmsd_before = float(np.sqrt(((ref_pts - mob_pts)**2).sum(axis=1).mean()))

    ref_cen = ref_pts.mean(axis=0)
    mob_cen = mob_pts.mean(axis=0)
    H   = (mob_pts - mob_cen).T @ (ref_pts - ref_cen)
    U, S, Vt = np.linalg.svd(H)
    d   = np.linalg.det(Vt.T @ U.T)
    rot = Vt.T @ np.diag([1, 1, d]) @ U.T
    t   = ref_cen - rot @ mob_cen

    mob_aligned = (rot @ mob_pts.T).T + t
    rmsd_after  = float(np.sqrt(((ref_pts - mob_aligned)**2).sum(axis=1).mean()))

    new_lines = []
    for line in mob_lines:
        if line.startswith(("ATOM","HETATM")):
            try:
                pos = np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])])
                pt  = rot @ pos + t
                new_lines.append(line[:30]+f"{pt[0]:8.3f}{pt[1]:8.3f}{pt[2]:8.3f}"+line[54:])
                continue
            except (ValueError, IndexError):
                pass
        new_lines.append(line)
    return new_lines, rmsd_before, rmsd_after


def calc_active_site_rmsd(ref_lines, mob_lines, site_residues) -> float:
    def parse_ca(lines):
        ca = {}
        for line in lines:
            if not line.startswith("ATOM"): continue
            if line[12:16].strip() != "CA": continue
            try:
                key = (line[21].strip(), line[22:26].strip())
                if key not in site_residues: continue
                ca[key] = np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])])
            except (ValueError, IndexError): continue
        return ca
    ref_ca = parse_ca(ref_lines)
    mob_ca = parse_ca(mob_lines)
    common = sorted(set(ref_ca.keys()) & set(mob_ca.keys()))
    if len(common) < 3: return -1.0
    ref_pts = np.array([ref_ca[k] for k in common])
    mob_pts = np.array([mob_ca[k] for k in common])
    return float(np.sqrt(((ref_pts - mob_pts)**2).sum(axis=1).mean()))


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: PDBFixer (verbatim from working monolith)
# ══════════════════════════════════════════════════════════════════════════════

# Water residue names to always consider as solvent
_WATER_NAMES = {"HOH", "WAT", "SOL", "H2O", "TIP", "TIP3", "TIP4"}


def fix_receptor(pdb_path: Path, out_dir: Path,
                 ph: float = 7.4,
                 remove_resnames: list[str] | None = None,
                 keep_water: bool = False) -> Path:
    """
    Fix receptor structure with PDBFixer.

    remove_resnames: specific HETATM residue names to remove (e.g. ["LIG"]).
                     If None, all HETATM records are removed (default).
                     If empty list [], no HETATM records are removed —
                     cofactors, metals, and ligands are all kept.
    keep_water:      if True, water molecules are retained.

    Strategy: save HETATM records from the input PDB before running
    PDBFixer (which always strips them), then restore the ones we want
    to keep after PDBFixer processes the protein backbone.
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    fixed_pdb = out_dir / "receptor_fixed.pdb"
    print(f"  Fixing structure with PDBFixer (pH {ph}) …")

    # ── Save HETATM records from input before PDBFixer strips them ────────────
    saved_hetatm = []
    if remove_resnames is not None:
        # We'll need to restore some — save them all now
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("HETATM"):
                    continue
                rname = line[17:20].strip().upper()
                # Skip water unless keep_water
                if rname in _WATER_NAMES and not keep_water:
                    continue
                # Skip residues in the remove list
                if remove_resnames and rname in {r.upper() for r in remove_resnames}:
                    continue
                if not line.endswith("\n"): line = line + "\n"
                saved_hetatm.append(line)

    # ── Run PDBFixer on protein only ──────────────────────────────────────────
    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    # Always strip heterogens here — we restore what we want manually below
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    with open(fixed_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    # ── Post-process: apply HETATM policy ────────────────────────────────────
    lines = fixed_pdb.read_text().splitlines(keepends=True)

    # Strip any HETATM PDBFixer may have written (shouldn't be any, but safe)
    lines = [l for l in lines if not l.startswith("HETATM")]

    if remove_resnames is None:
        # Default: keep no HETATM (protein-only receptor)
        n_removed = len(saved_hetatm)
        if n_removed:
            print(f"  Removed all {n_removed} HETATM record(s)")
    else:
        # Restore saved HETATM records (filtered above to exclude remove_resnames)
        # Insert before the END record
        end_idx = next((i for i, l in enumerate(lines)
                        if l.startswith("END")), len(lines))
        lines = lines[:end_idx] + saved_hetatm + lines[end_idx:]
        n_kept = len(saved_hetatm)
        if n_kept:
            kept_names = sorted({l[17:20].strip() for l in saved_hetatm})
            print(f"  Retained {n_kept} HETATM record(s): "
                  f"{', '.join(kept_names)}")
        if remove_resnames:
            print(f"  Removed HETATM resname(s): "
                  f"{', '.join(remove_resnames)}")

    fixed_pdb.write_text("".join(lines))
    n_atoms = sum(1 for l in lines if l.startswith(("ATOM","HETATM")))
    print(f"  Fixed PDB: {fixed_pdb.name} ({n_atoms} atoms)")
    return fixed_pdb


# ══════════════════════════════════════════════════════════════════════════════
# Step 2a: PDBQT conversion (verbatim from working monolith)
# ══════════════════════════════════════════════════════════════════════════════

def make_receptor_pdbqt(fixed_pdb: Path, out_dir: Path) -> Path:
    rigid_pdbqt = out_dir / "receptor_rigid.pdbqt"

    # Check fixed PDB has content before attempting conversion
    all_lines = fixed_pdb.read_text().splitlines(keepends=True)
    pdb_lines = [l for l in all_lines if l.startswith("ATOM")]
    if not pdb_lines:
        raise RuntimeError(f"Fixed PDB has no ATOM records: {fixed_pdb}")

    # Write a protein-only PDB for PDBQT conversion.
    # HETATM records (cofactors, metals) must be excluded because:
    # 1. Gasteiger charge assignment fails for metal ions (Ca, Mg, Zn etc.)
    #    causing autogrid4 to error with "no partial charges found"
    # 2. Cofactor atoms don't require separate affinity maps for docking
    # The original receptor_fixed.pdb (with cofactors) is still used for
    # complex building after docking.
    n_hetatm = sum(1 for l in all_lines if l.startswith("HETATM"))
    if n_hetatm > 0:
        protein_only_pdb = out_dir / "_protein_only.pdb"
        protein_only_pdb.write_text("".join(
            l for l in all_lines if not l.startswith("HETATM")
        ))
        conversion_pdb = protein_only_pdb
        print(f"  PDBQT conversion: excluding {n_hetatm} HETATM record(s) "
              f"(cofactors retained in fixed PDB for complex building)")
    else:
        conversion_pdb = fixed_pdb

    # Try obabel CLI first
    # Use OpenBabel Python API directly — more reliable charge assignment
    # than the CLI for receptor PDBQT generation.
    pdbqt_ok = False
    try:
        from openbabel import openbabel as ob
        ob.obErrorLog.SetOutputLevel(ob.obError)
        conv = ob.OBConversion()
        conv.SetInAndOutFormats("pdb", "pdbqt")
        conv.AddOption("r", ob.OBConversion.OUTOPTIONS)
        mol = ob.OBMol()
        if not conv.ReadFile(mol, str(conversion_pdb)):
            raise RuntimeError("OpenBabel Python API: cannot read fixed PDB")
        charge_model = ob.OBChargeModel.FindType("gasteiger")
        if charge_model:
            charge_model.ComputeCharges(mol)
        else:
            raise RuntimeError("Gasteiger charge model not available")
        if not conv.WriteFile(mol, str(rigid_pdbqt)):
            raise RuntimeError("OpenBabel Python API: cannot write PDBQT")
        # Verify charges were assigned
        lines = rigid_pdbqt.read_text().splitlines()
        atom_lines = [l for l in lines if l.startswith("ATOM")]
        charges = []
        for l in atom_lines:
            try:
                charges.append(float(l[70:76]))
            except (ValueError, IndexError):
                pass
        nonzero = sum(1 for c in charges if abs(c) > 0.001)
        if nonzero == 0:
            raise RuntimeError("All charges are zero after Gasteiger assignment")
        pdbqt_ok = bool(atom_lines)
        print(f"  Charges: {nonzero}/{len(charges)} atoms have non-zero Gasteiger charge")
    except Exception as e:
        print(f"  OpenBabel Python API failed: {e} — trying CLI fallback")
        result = subprocess.run(
            ["obabel", str(conversion_pdb), "-O", str(rigid_pdbqt),
             "-xr", "--partialcharge", "gasteiger"],
            capture_output=True, text=True
        )
        pdbqt_ok = (rigid_pdbqt.exists() and
                    any(l.startswith("ATOM") for l in
                        rigid_pdbqt.read_text().splitlines()))

    if not pdbqt_ok:
        print(f"  Both charge methods failed — trying minimal PDBQT fallback")

    if not pdbqt_ok:
        # Last resort: write minimal PDBQT directly from fixed PDB ATOM records
        # with zero charges — sufficient for autogrid4 map generation
        print(f"  Writing minimal PDBQT from fixed PDB ATOM records")
        out_lines = []
        for line in pdb_lines:
            if len(line) < 54: continue
            aname = line[12:16].strip()
            elem  = line[76:78].strip() if len(line) > 76 else ""
            if not elem:
                raw  = "".join(c for c in aname if c.isalpha())
                elem = (raw[0].upper() + raw[1:].lower()
                        if len(raw) > 1 else raw.upper())
            # Map element to AD4 type
            ad4 = {"C":"C","N":"N","O":"OA","S":"S","H":"HD",
                   "P":"P","F":"F"}.get(elem[0].upper(), "C")
            out_lines.append(line.rstrip()[:54].ljust(54) +
                             "  1.00  0.00     0.000 " + ad4)
        out_lines.append("END")
        rigid_pdbqt.write_text("\n".join(out_lines) + "\n")
        pdbqt_ok = len(out_lines) > 1

    if not pdbqt_ok:
        raise RuntimeError("All PDBQT conversion methods failed")

    # Clean up temp protein-only PDB if it was created
    try:
        (out_dir / "_protein_only.pdb").unlink(missing_ok=True)
    except Exception:
        pass

    n_atoms = sum(1 for l in rigid_pdbqt.read_text().splitlines()
                  if l.startswith("ATOM"))
    print(f"  Receptor PDBQT: {rigid_pdbqt.name} ({n_atoms} atoms)")
    return rigid_pdbqt


# ══════════════════════════════════════════════════════════════════════════════
# Step 2b: Flex receptor split (verbatim from working monolith)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_pdb_residues(pdb_path: Path) -> dict:
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM","HETATM")): continue
            try:
                x,y,z = float(line[30:38]),float(line[38:46]),float(line[46:54])
            except ValueError: continue
            key = (line[21].strip(), line[17:20].strip(), line[22:26].strip())
            residues.setdefault(key,{})[line[12:16].strip()] = (x,y,z,line)
    return residues


def _in_box(x, y, z, box) -> bool:
    return (abs(x-box["cx"]) <= box["sx"]/2 and
            abs(y-box["cy"]) <= box["sy"]/2 and
            abs(z-box["cz"]) <= box["sz"]/2)


def split_flex_receptor(fixed_pdb: Path, rigid_pdbqt: Path,
                         out_dir: Path, box: dict,
                         max_flex: int = 12, max_lig_bonds: int = 30):
    residues = _parse_pdb_residues(fixed_pdb)
    flex_keys = []
    for key, atoms in residues.items():
        chain, resname, resseq = key
        if resname not in _SC_BONDS or not _SC_BONDS.get(resname, True): continue
        for aname, (x,y,z,_) in atoms.items():
            if aname not in _BACKBONE and _in_box(x,y,z,box):
                flex_keys.append(key); break

    if not flex_keys:
        print(f"  No sidechain atoms inside box — rigid receptor only")
        return rigid_pdbqt, None

    cx,cy,cz = box["cx"],box["cy"],box["cz"]
    def _d(key):
        pts = list(residues[key].values())
        xs=[v[0] for v in pts]; ys=[v[1] for v in pts]; zs=[v[2] for v in pts]
        return ((sum(xs)/len(xs)-cx)**2+(sum(ys)/len(ys)-cy)**2+(sum(zs)/len(zs)-cz)**2)**0.5
    flex_keys = sorted(flex_keys, key=_d)

    max_flex_bonds = 128 - max_lig_bonds
    selected, total_bonds = [], 0
    for key in flex_keys:
        if len(selected) >= max_flex: break
        res_bonds = len(_SC_BONDS.get(key[1], []))
        if total_bonds + res_bonds <= max_flex_bonds:
            selected.append(key); total_bonds += res_bonds
    flex_keys = selected
    print(f"  Selected {len(flex_keys)} flex residue(s): "
          f"{total_bonds}/{max_flex_bonds} torsion bonds used "
          f"({max_lig_bonds} reserved for ligand)")

    if not flex_keys:
        print(f"  No residues fit within bond budget — rigid only")
        return rigid_pdbqt, None

    flex_set = {(c,r,s) for c,r,s in flex_keys}
    flex_pdbqt = out_dir / "receptor_flex.pdbqt"

    for chain,resname,resseq in flex_keys:
        print(f"    Chain {chain}  {resname:<4s} {resseq:>4s}  "
              f"({len(_SC_BONDS.get(resname,[]))} rotatable bond(s))")

    new_rigid_pdbqt = out_dir / "receptor_rigid_split.pdbqt"
    lines = rigid_pdbqt.read_text().splitlines()
    rigid_lines = []
    for line in lines:
        if line.startswith(("ATOM","HETATM")):
            key = (line[21].strip(), line[17:20].strip(), line[22:26].strip())
            if key in flex_set and line[12:16].strip() not in _BACKBONE: continue
        rigid_lines.append(line)
    new_rigid_pdbqt.write_text("\n".join(rigid_lines)+"\n")

    serial = [1]; atom_serials = {}

    def _vtype(aname):
        s=aname.lstrip("0123456789"); e=s[0] if s else "C"
        if e=="C": return "A" if aname in {"CG","CD","CE","CZ","CD1","CD2","CE1","CE2","CZ2","CZ3","CH2"} else "C"
        if e=="N": return "NA" if aname in {"NE","NH1","NH2","ND1","ND2","NE1","NE2","NZ"} else "N"
        if e=="O": return "OA"
        if e=="S": return "SA" if aname=="SG" else "S"
        return e

    def _atom_line(aname, atoms):
        if aname not in atoms: return None
        x,y,z,orig = atoms[aname]
        s=serial[0]; serial[0]+=1; atom_serials[aname]=s
        res = orig[17:27] if len(orig)>=27 else orig[17:].ljust(10)
        return f"ATOM  {s:5d} {aname:<4s} {res}   {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00     0.000 {_vtype(aname)} "

    flex_blocks = []
    for chain,resname,resseq in flex_keys:
        atoms = residues[(chain,resname,resseq)]
        bonds = _SC_BONDS.get(resname,[])
        if not bonds or "CA" not in atoms: continue
        serial[0]=1; atom_serials.clear()
        chain_atoms = {a for pair in bonds for a in pair}
        written = set(); lines_out = []
        ca = _atom_line("CA", atoms)
        if not ca: continue
        lines_out += ["ROOT", ca, "ENDROOT"]; written.add("CA")
        open_b = []
        for pn,cn in bonds:
            while open_b and open_b[-1][1]!=pn:
                op,oc = open_b.pop()
                for an in atoms:
                    if an not in written and an not in _BACKBONE and an not in chain_atoms:
                        al=_atom_line(an,atoms)
                        if al: lines_out.append(al); written.add(an)
                lines_out.append(f"ENDBRANCH {atom_serials[op]:3d} {atom_serials[oc]:3d}")
            if pn not in atom_serials or cn not in atoms: continue
            cl = _atom_line(cn, atoms)
            if not cl: continue
            lines_out.append(f"BRANCH {atom_serials[pn]:3d} {atom_serials[cn]:3d}")
            lines_out.append(cl); written.add(cn); open_b.append((pn,cn))
        while open_b:
            op,oc = open_b.pop()
            for an in atoms:
                if an not in written and an not in _BACKBONE:
                    al=_atom_line(an,atoms)
                    if al: lines_out.append(al); written.add(an)
            lines_out.append(f"ENDBRANCH {atom_serials[op]:3d} {atom_serials[oc]:3d}")
        if not any(l.startswith("BRANCH") for l in lines_out): continue
        flex_blocks.append(
            f"BEGIN_RES {resname} {chain} {resseq}\n"
            + "\n".join(lines_out)+"\n"
            + f"END_RES {resname} {chain} {resseq}"
        )

    if not flex_blocks:
        print(f"  No valid flex blocks generated — rigid only")
        return rigid_pdbqt, None

    flex_pdbqt.write_text("\n".join(flex_blocks)+"\n")
    print(f"  Rigid PDBQT: {new_rigid_pdbqt.name}")
    print(f"  Flex  PDBQT: {flex_pdbqt.name}")
    return new_rigid_pdbqt, flex_pdbqt


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: autogrid4 map generation (verbatim from working monolith)
# ══════════════════════════════════════════════════════════════════════════════

def generate_maps(rigid_pdbqt: Path, maps_dir: Path, box: dict,
                  spacing: float = 0.375, atom_types: list = None) -> Path:
    if atom_types is None:
        atom_types = _DEFAULT_AD4_TYPES

    if maps_dir.exists():
        shutil.rmtree(maps_dir)
    maps_dir.mkdir(parents=True, exist_ok=True)

    receptor_local = maps_dir / "receptor.pdbqt"
    shutil.copy(rigid_pdbqt, receptor_local)

    ad4_param = None
    for candidate in [
        "/usr/share/autodock/AD4_parameters.dat",
        "/usr/local/share/autodock/AD4_parameters.dat",
        "/usr/lib/autodock/AD4_parameters.dat",
        "/app/AD4_parameters.dat",
    ]:
        if Path(candidate).exists():
            ad4_param = Path(candidate); break

    if ad4_param is None:
        result = subprocess.run(
            ["find", "/", "-name", "AD4_parameters.dat", "-type", "f"],
            capture_output=True, text=True)
        found = result.stdout.strip().splitlines()
        if found: ad4_param = Path(found[0])

    if ad4_param is None:
        raise RuntimeError("AD4_parameters.dat not found in container.")

    shutil.copy(ad4_param, maps_dir / "AD4_parameters.dat")
    print(f"  AD4 parameters: {ad4_param}")

    def _npts(size, spacing):
        n = int(size / spacing)
        return n if n % 2 == 0 else n + 1

    npts_x = _npts(box["sx"], spacing)
    npts_y = _npts(box["sy"], spacing)
    npts_z = _npts(box["sz"], spacing)

    types_str = " ".join(atom_types)
    gpf_lines = [
        f"npts {npts_x} {npts_y} {npts_z}",
        "parameter_file AD4_parameters.dat",
        "gridfld receptor.maps.fld",
        f"spacing {spacing}",
        "receptor_types A C HD NA N OA SA S",
        f"ligand_types {types_str}",
        "receptor receptor.pdbqt",
        f"gridcenter {box['cx']:.4f} {box['cy']:.4f} {box['cz']:.4f}",
        "smooth 0.5",
    ]
    for t in atom_types:
        gpf_lines.append(f"map receptor.{t}.map")
    gpf_lines += [
        "elecmap receptor.e.map",
        "dsolvmap receptor.d.map",
        "dielectric -0.1465",
        "",
    ]

    gpf_content = "\n".join(gpf_lines)
    gpf_path = maps_dir / "receptor.gpf"
    gpf_path.write_text(gpf_content)

    result = subprocess.run(
        ["autogrid4", "-p", "receptor.gpf", "-l", "receptor.glg"],
        capture_output=True, text=True, cwd=str(maps_dir)
    )
    if result.returncode != 0:
        print(f"  autogrid4 stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"autogrid4 failed (exit {result.returncode})")

    fld_path = maps_dir / "receptor.maps.fld"
    if not fld_path.exists():
        raise RuntimeError(f"autogrid4 did not produce {fld_path.name}")

    n_maps = len(list(maps_dir.glob("*.map")))
    print(f"  Maps ready: {fld_path.name} ({n_maps} .map files)")
    return fld_path


# ══════════════════════════════════════════════════════════════════════════════
# Library helpers
# ══════════════════════════════════════════════════════════════════════════════

def autobox_from_library(library_dir: Path, padding: float = 10.0):
    db_path = library_dir / "library_state.db"
    if db_path.exists():
        try:
            import sqlite3
            con  = sqlite3.connect(str(db_path), timeout=10)
            rows = con.execute(
                "SELECT bbox_json FROM ligands WHERE status='ready' AND bbox_json IS NOT NULL"
            ).fetchall()
            con.close()
            if rows:
                bboxes = [json.loads(r[0]) for r in rows]
                return (round(max(b["dx"] for b in bboxes)+2*padding,1),
                        round(max(b["dy"] for b in bboxes)+2*padding,1),
                        round(max(b["dz"] for b in bboxes)+2*padding,1))
        except Exception: pass
    json_path = library_dir / "library.json"
    if json_path.exists():
        try:
            lib = json.loads(json_path.read_text())
            ok  = [v for v in lib.values() if v.get("status")=="ok" and v.get("bbox")]
            if ok:
                return (round(max(e["bbox"]["dx"] for e in ok)+2*padding,1),
                        round(max(e["bbox"]["dy"] for e in ok)+2*padding,1),
                        round(max(e["bbox"]["dz"] for e in ok)+2*padding,1))
        except Exception: pass
    return None


def ad4_types_from_library(library_dir: Path) -> list[str]:
    elem_to_ad4 = {
        "C":["A","C"],"N":["N","NA"],"O":["OA"],"S":["S","SA"],"H":["HD"],
        "P":["P"],"F":["F"],"Cl":["Cl"],"Br":["Br"],"I":["I"],
    }
    all_elems = set()
    db_path = library_dir / "library_state.db"
    if db_path.exists():
        try:
            import sqlite3
            con  = sqlite3.connect(str(db_path), timeout=10)
            rows = con.execute(
                "SELECT elements_json FROM ligands WHERE status='ready' AND elements_json IS NOT NULL"
            ).fetchall()
            con.close()
            for (ej,) in rows:
                all_elems.update(json.loads(ej))
        except Exception: pass
    if not all_elems:
        json_path = library_dir / "library.json"
        if json_path.exists():
            try:
                lib = json.loads(json_path.read_text())
                for v in lib.values():
                    all_elems.update(v.get("elements",[]))
            except Exception: pass
    if not all_elems:
        return _DEFAULT_AD4_TYPES
    types = {"HD"}
    for e in all_elems:
        types.update(elem_to_ad4.get(e,[]))
    result = [t for t in _AD4_TYPE_PRIORITY if t in types][:12]
    return result or _DEFAULT_AD4_TYPES


def max_rot_from_library(library_dir: Path) -> int:
    db_path = library_dir / "library_state.db"
    if db_path.exists():
        try:
            import sqlite3
            con = sqlite3.connect(str(db_path), timeout=10)
            row = con.execute("SELECT MAX(n_rot_bonds) FROM ligands WHERE status='ready'").fetchone()
            con.close()
            if row and row[0]: return int(row[0])
        except Exception: pass
    return 30


# ══════════════════════════════════════════════════════════════════════════════
# Full model preparation
# ══════════════════════════════════════════════════════════════════════════════

def prep_model(model_lines: list[str], model_dir: Path, box: dict,
               ad4_types: list[str], max_lig_bonds: int,
               ph: float, spacing: float,
               remove_resnames: list[str] | None = None,
               keep_water: bool = False,
               transform=None, core_mask=None, seed_info=None) -> dict:
    model_dir.mkdir(parents=True, exist_ok=True)
    tmp_pdb = model_dir / "_input.pdb"
    tmp_pdb.write_text("".join(model_lines))
    try:
        fixed_pdb   = fix_receptor(tmp_pdb, model_dir, ph,
                                    remove_resnames, keep_water)
        rigid_pdbqt = make_receptor_pdbqt(fixed_pdb, model_dir)
        rigid_pdbqt, flex_pdbqt = split_flex_receptor(
            fixed_pdb, rigid_pdbqt, model_dir, box,
            max_lig_bonds=max_lig_bonds)
        fld_file = generate_maps(rigid_pdbqt, model_dir / "maps",
                                  box, spacing, ad4_types)
        meta = {
            "fld_file":    str(fld_file.resolve()),
            "fixed_pdb":   str(fixed_pdb.resolve()),
            "rigid_pdbqt": str(rigid_pdbqt.resolve()),
            "flex_pdbqt":  str(flex_pdbqt.resolve()) if flex_pdbqt else None,
            "flex_enabled": flex_pdbqt is not None,
            "ad4_types":   ad4_types,
            "box":         box,
            "transform":   transform,
            "core_mask":   core_mask,
            "seed":        seed_info,
        }
        (model_dir / "receptor_prep.json").write_text(json.dumps(meta, indent=2))
        return meta
    finally:
        tmp_pdb.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main processing
# ══════════════════════════════════════════════════════════════════════════════

def _measure_cavity_box(directive: dict, ref_lines: list[str], orig_pdb: Path,
                        receptor_dir: Path, ph: float,
                        remove_resnames: list[str] | None, keep_water: bool,
                        spacing: float):
    """
    Measure the cavity box on the reference model.

    Input:  directive - the cavity box dict from main() (seed + options)
            ref_lines - cleaned reference-model PDB lines
            orig_pdb  - uploaded PDB (for a native-ligand seed)
            receptor_dir, ph, remove_resnames, keep_water, spacing - prep params
    Output: (box {cx,cy,cz,sx,sy,sz}, transform 4x4, core_mask, seed_info)
    Behaviour: fixes the reference, builds a rigid PDBQT, resolves the seed, and
               calls cavity_box.measure_box (which runs a throwaway provisional
               AutoGrid). Reorientation is enabled only when the directive asks;
               otherwise the transform is identity and the box stays in the
               uploaded frame. The scratch directory is removed afterwards.
    """
    aux = receptor_dir / "_cavity_prep"
    if aux.exists():
        shutil.rmtree(aux)
    aux.mkdir(parents=True, exist_ok=True)
    try:
        tmp_pdb = aux / "_ref.pdb"
        tmp_pdb.write_text("".join(ref_lines))
        fixed = fix_receptor(tmp_pdb, aux, ph, remove_resnames, keep_water)
        rigid = make_receptor_pdbqt(fixed, aux)
        seed, source = cavity_box.resolve_seed(orig_pdb, fixed, directive)
        # Reorientation is computed-ready in cavity_box, but applying it means
        # rotating the receptor and inverse-transforming docked poses (the full
        # transform-bookkeeping step). Until that lands and is validated, keep
        # every receptor in the uploaded frame so the box can never end up in a
        # frame the receptor is not in.
        if directive.get("reorient"):
            print("  NOTE: reorientation requested but not yet applied at prep; "
                  "keeping the uploaded frame (box is still cavity-fitted)")
        reorient_gain = float("inf")
        measured = cavity_box.measure_box(
            rigid, seed, spacing, generate_maps, aux / "prov",
            provisional_size=directive.get("provisional_size", 30.0),
            apron_reach=directive.get("apron_reach", 8.0),
            reorient_gain=reorient_gain)
        seed_info = {"xyz": measured["seed"], "source": source,
                     "reoriented": measured["reoriented"], "gain": measured["gain"]}

        # Prep-time sanity: a native ligand must sit inside the grown core, or
        # the burial threshold / structure is wrong and the box is suspect.
        native = cavity_box.native_ligand_atoms(orig_pdb,
                                                directive.get("native_resname"))
        if len(native):
            native_f = cavity_box.apply_transform(
                native, np.array(measured["transform"], dtype=float))
            frac = cavity_box.fraction_in_core(native_f, measured["core_mask"])
            seed_info["native_in_core"] = round(frac, 3)
            if frac < 0.5:
                print(f"  WARNING: native ligand only {frac:.0%} inside the grown "
                      f"core — burial threshold or structure may be off; the box "
                      f"may be mis-centred")
            else:
                print(f"  Native-ligand check: {frac:.0%} inside the core")

        return (dict(measured["box"]), measured["transform"],
                measured["core_mask"], seed_info)
    finally:
        shutil.rmtree(aux, ignore_errors=True)


def process_receptor(pdb_path: Path, receptor_dir: Path, box: dict,
                     ad4_types: list[str], max_lig_bonds: int = 30,
                     rmsd_threshold: float = 0.5, ph: float = 7.4,
                     spacing: float = 0.375,
                     remove_resnames: list[str] | None = None,
                     keep_water: bool = False) -> bool:
    state      = ReceptorState(receptor_dir)
    models_dir = receptor_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    raw_models = split_models(pdb_path)
    print(f"\n  Models found: {len(raw_models)}")

    ref_lines     = None
    site_residues = None
    ensemble      = []
    transform     = None
    core_mask     = None
    seed_info     = None
    n_ok = n_excluded = n_failed = 0

    for model_num, model_lines in raw_models:
        clean_lines = strip_model_records(model_lines)
        if not any(l.startswith("ATOM") for l in clean_lines):
            print(f"  Model {model_num}: no ATOM records — skipping")
            n_failed += 1; continue

        if ref_lines is None:
            ref_lines = clean_lines
            seq = extract_sequence(clean_lines)
            (receptor_dir / "receptor_sequence.txt").write_text(seq)
            state.set_info(sequence_fingerprint=seq)
            aligned_lines = clean_lines
            rmsd_before = rmsd_after = as_rmsd = 0.0
            print(f"  Model {model_num}: reference")
            if isinstance(box, dict) and box.get("mode") == "cavity":
                try:
                    box, transform, core_mask, seed_info = _measure_cavity_box(
                        box, clean_lines, pdb_path, receptor_dir, ph,
                        remove_resnames, keep_water, spacing)
                    print(f"  Cavity box: centre ({box['cx']:.2f}, {box['cy']:.2f}, "
                          f"{box['cz']:.2f})  size {box['sx']} × {box['sy']} × "
                          f"{box['sz']} Å  (seed: {seed_info['source']})")
                except Exception as e:
                    print(f"  ERROR: cavity box measurement failed — {e}")
                    return False
        else:
            if site_residues is None:
                site_residues = get_active_site_residues(ref_lines, box)
                print(f"  Active site: {len(site_residues)} residues within box+2Å")

            aligned_lines, rmsd_before, rmsd_after = superpose_to_reference(
                ref_lines, clean_lines, model_num, site_residues)
            as_rmsd = calc_active_site_rmsd(ref_lines, aligned_lines, site_residues)
            print(f"  Model {model_num}: active site RMSD {as_rmsd:.3f} Å "
                  f"(full Cα: {rmsd_before:.2f}→{rmsd_after:.2f} Å)")

            if 0 <= as_rmsd < rmsd_threshold:
                print(f"  Model {model_num}: EXCLUDED — too similar to model 1")
                n_excluded += 1; continue

        model_dir = models_dir / f"model_{model_num:03d}"
        state.add_model(model_num, str(model_dir), str(pdb_path))
        if not state.claim_model(model_num):
            print(f"  Model {model_num}: already processing"); n_ok += 1; continue

        print(f"\n  --- Model {model_num}/{len(raw_models)} ---")
        try:
            as_val = as_rmsd if ref_lines is not aligned_lines else -1.0
            meta   = prep_model(aligned_lines, model_dir, box,
                                 ad4_types, max_lig_bonds, ph, spacing,
                                 remove_resnames, keep_water,
                                 transform=transform, core_mask=core_mask,
                                 seed_info=seed_info)
            state.mark_model_ready(model_num, as_val)

            ensemble.append({
                "model_id":         model_num,
                "prep_dir":         str(model_dir.resolve()),
                "fld_file":         meta["fld_file"],
                "active_site_rmsd": as_val,
            })
            n_ok += 1
            print(f"  Model {model_num}: ready")
        except Exception as e:
            import traceback; traceback.print_exc()
            state.mark_model_failed(model_num, str(e))
            print(f"  Model {model_num}: FAILED — {e}")
            n_failed += 1

    ensemble_path = receptor_dir / "ensemble.json"
    ensemble_data = {
        "receptor_name":   receptor_dir.name,
        "source_pdb":      str(pdb_path.resolve()),
        "box":             box,
        "transform":       transform,
        "core_mask":       core_mask,
        "seed":            seed_info,
        "n_models":        len(raw_models),
        "n_ready":         n_ok,
        "n_excluded":      n_excluded,
        "n_failed":        n_failed,
        "rmsd_threshold":  rmsd_threshold,
        "conformations":   ensemble,
    }
    ensemble_path.write_text(json.dumps(ensemble_data, indent=2))

    # Write receptor_metadata.json (persistent record of this receptor)
    metadata_path = receptor_dir / "receptor_metadata.json"
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps({
            "receptor_name": receptor_dir.name,
            "display_name":  receptor_dir.name,
            "variant":       "",
            "gene_id":       "",
            "gene_db":       "",
            "source_pdb":    str(pdb_path.resolve()),
            "prepped_at":    __import__("datetime").datetime.now().isoformat(),
            "notes":         "",
        }, indent=2))

    print(f"\n  === Summary ===")
    print(f"  Ready:    {n_ok}")
    print(f"  Excluded: {n_excluded} (active site RMSD < {rmsd_threshold} Å)")
    print(f"  Failed:   {n_failed}")
    print(f"  Ensemble: {ensemble_path}")

    state.set_info(n_models=len(raw_models), n_ready=n_ok,
                   n_excluded=n_excluded, n_failed=n_failed)
    return n_ok > 0


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Prepare receptor PDB(s) for docking.")
    p.add_argument("--job",            help="Job JSON file (preferred — all params inside)")
    p.add_argument("--pdb",            help="Input PDB (from inbox)")
    p.add_argument("--receptor_dir",   help="Existing receptor directory (for rerun)")
    p.add_argument("--receptors_dir",  default="/receptors")
    p.add_argument("--library_dir",    default="/library")
    p.add_argument("--name",           help="Receptor name")
    p.add_argument("--center_x",       type=float, default=None)
    p.add_argument("--center_y",       type=float, default=None)
    p.add_argument("--center_z",       type=float, default=None)
    p.add_argument("--size_x",         type=float, default=None)
    p.add_argument("--size_y",         type=float, default=None)
    p.add_argument("--size_z",         type=float, default=None)
    p.add_argument("--autobox",        action="store_true")
    p.add_argument("--padding",        type=float, default=10.0)
    p.add_argument("--spacing",        type=float, default=0.375)
    p.add_argument("--ph",             type=float, default=7.4)
    p.add_argument("--rmsd_threshold", type=float, default=0.5)
    p.add_argument("--remove_resnames", nargs="*", default=None,
                   metavar="RESNAME",
                   help="HETATM residue names to remove (e.g. LIG ATP). "
                        "Omit to remove all HETATM. Pass empty to keep all.")
    p.add_argument("--keep_water",     action="store_true",
                   help="Retain water molecules after PDBFixer")
    p.add_argument("--rerun",          action="store_true")
    p.add_argument("--box_mode",       choices=["explicit", "cavity"], default=None,
                   help="'cavity' measures the box from the receptor pocket")
    p.add_argument("--site_residues",  nargs="*", default=None, metavar="RES",
                   help="Residues (CHAIN:RESSEQ or RESSEQ) whose centroid seeds "
                        "the cavity search")
    p.add_argument("--native_resname", default=None,
                   help="HETATM resname to use as the native-ligand seed")
    p.add_argument("--apron_reach",    type=float, default=8.0,
                   help="Surface apron reach for the cavity box (0 disables)")
    p.add_argument("--reorient",       action="store_true",
                   help="Allow receptor reorientation (applied in a later step)")
    p.add_argument("--reorient_gain",  type=float, default=2.0,
                   help="Reorient only if it shrinks the box by >= this factor")
    p.add_argument("--provisional_size", type=float, default=30.0,
                   help="Edge of the throwaway provisional grid around the seed")
    args = p.parse_args()

    # If job file provided, overlay its values onto args
    job_data = {}
    if args.job:
        with open(args.job) as f:
            job_data = json.load(f)
        opts = job_data.get("options", {})
        meta = job_data.get("metadata", {})
        if job_data.get("pdb_path"):     args.pdb           = job_data["pdb_path"]
        if job_data.get("receptors_dir"): args.receptors_dir = job_data["receptors_dir"]
        if job_data.get("library_dir"):  args.library_dir   = job_data["library_dir"]
        if opts.get("center_x") is not None: args.center_x  = float(opts["center_x"])
        if opts.get("center_y") is not None: args.center_y  = float(opts["center_y"])
        if opts.get("center_z") is not None: args.center_z  = float(opts["center_z"])
        if opts.get("padding")  is not None: args.padding    = float(opts["padding"])
        if opts.get("rmsd_threshold") is not None:
            args.rmsd_threshold = float(opts["rmsd_threshold"])
        if opts.get("size_x") is not None: args.size_x = float(opts["size_x"])
        if opts.get("size_y") is not None: args.size_y = float(opts["size_y"])
        if opts.get("size_z") is not None: args.size_z = float(opts["size_z"])
        if "remove_resnames" in opts:      args.remove_resnames = opts["remove_resnames"]
        if opts.get("keep_water"):         args.keep_water = True
        if opts.get("box_mode"):           args.box_mode = opts["box_mode"]
        if opts.get("site_residues") is not None:
            args.site_residues = opts["site_residues"]
        if opts.get("native_resname"):     args.native_resname = opts["native_resname"]
        if opts.get("apron_reach") is not None:
            args.apron_reach = float(opts["apron_reach"])
        if opts.get("reorient"):           args.reorient = True
        if opts.get("reorient_gain") is not None:
            args.reorient_gain = float(opts["reorient_gain"])
        if opts.get("provisional_size") is not None:
            args.provisional_size = float(opts["provisional_size"])
        if meta.get("name"): args.name = meta["name"]
        args.autobox = True

    library_dir   = Path(args.library_dir)
    receptors_dir = Path(args.receptors_dir)

    # Box size — prefer explicit values from job file (computed by webui
    # which has direct library DB access; containers cannot reliably read
    # the WAL-mode library DB). Fall back to autobox for CLI use.
    if args.box_mode == "cavity":
        # Box is measured from the receptor pocket on the reference model inside
        # process_receptor; carry the seed and cavity options forward.
        box = {
            "mode":             "cavity",
            "center_x":         args.center_x,
            "center_y":         args.center_y,
            "center_z":         args.center_z,
            "site_residues":    args.site_residues,
            "native_resname":   args.native_resname,
            "apron_reach":      args.apron_reach,
            "reorient":         args.reorient,
            "reorient_gain":    args.reorient_gain,
            "provisional_size": args.provisional_size,
        }
        print(f"  Box mode: cavity-measured (seed + box resolved at prep)")
    else:
        if args.size_x and args.size_y and args.size_z:
            sx, sy, sz = args.size_x, args.size_y, args.size_z
            print(f"  Box size: {sx} × {sy} × {sz} Å (from job file)")
        elif args.autobox:
            result = autobox_from_library(library_dir, args.padding)
            if result:
                sx, sy, sz = result
                print(f"  Autobox: {sx} × {sy} × {sz} Å (padding {args.padding} Å)")
            else:
                print("WARNING: Could not compute autobox — using 25×25×25 Å default")
                sx = sy = sz = 25.0
        else:
            print("ERROR: Provide --box_mode cavity, --size_x/y/z, or --autobox",
                  file=sys.stderr)
            sys.exit(1)

        box = {"cx": args.center_x, "cy": args.center_y, "cz": args.center_z,
               "sx": sx, "sy": sy, "sz": sz}

    ad4_types    = ad4_types_from_library(library_dir)
    max_lig_bonds = max_rot_from_library(library_dir)
    print(f"  AD4 types: {' '.join(ad4_types)}")
    print(f"  Flex bond budget: {128-max_lig_bonds} (128-{max_lig_bonds} for ligand)")

    # Resolve PDB and receptor directory
    if args.pdb:
        pdb_path = Path(args.pdb)
        if not pdb_path.exists():
            print(f"ERROR: PDB not found: {pdb_path}", file=sys.stderr); sys.exit(1)

        # Identity check
        with open(pdb_path) as f:
            pdb_lines = f.readlines()
        sequence = extract_sequence(pdb_lines)
        match_type, match_name, notes = match_receptor(sequence, receptors_dir)
        print(f"\n=== RECEPTOR INTAKE ===")
        print(f"  PDB:   {pdb_path.name}")
        print(f"  Match: {match_type} — {notes}")

        if args.name:
            # User-provided name takes priority
            receptor_name = args.name
            receptor_dir  = receptors_dir / receptor_name
        elif match_type == "exact":
            receptor_dir  = receptors_dir / match_name
            receptor_name = match_name
            print(f"  → Adding models to existing receptor: {match_name}")
        else:
            receptor_name = suggest_name(pdb_path, receptors_dir)
            receptor_dir  = receptors_dir / receptor_name
            print(f"  → New receptor: {receptor_name}")

        receptor_dir.mkdir(parents=True, exist_ok=True)
        orig = receptor_dir / "receptor_original.pdb"
        if not orig.exists() or args.rerun:
            shutil.copy2(pdb_path, orig)

    elif args.receptor_dir:
        receptor_dir = Path(args.receptor_dir)
        orig = receptor_dir / "receptor_original.pdb"
        if not orig.exists():
            print(f"ERROR: receptor_original.pdb not found", file=sys.stderr); sys.exit(1)
        pdb_path = orig
    else:
        print("ERROR: Provide --pdb or --receptor_dir", file=sys.stderr); sys.exit(1)

    print(f"\n=== RECEPTOR PREPARATION ===")
    print(f"  Name:    {receptor_dir.name}")
    if isinstance(box, dict) and box.get("mode") == "cavity":
        _seeded = "explicit XYZ" if box.get("center_x") is not None \
            else "residues / native ligand"
        print(f"  Box:     cavity-measured at prep (seed from {_seeded})")
    else:
        print(f"  Box:     centre ({box['cx']:.3f}, {box['cy']:.3f}, {box['cz']:.3f})")
        print(f"           size   ({box['sx']} × {box['sy']} × {box['sz']}) Å")

    if args.remove_resnames is not None:
        print(f"  Remove HETATM resnames: "
              f"{args.remove_resnames if args.remove_resnames else '(none — keep all)'}")
    else:
        print(f"  Remove HETATM: all")
    print(f"  Keep water: {args.keep_water}")

    ok = process_receptor(pdb_path, receptor_dir, box, ad4_types,
                          max_lig_bonds, args.rmsd_threshold, args.ph,
                          args.spacing, args.remove_resnames, args.keep_water)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
