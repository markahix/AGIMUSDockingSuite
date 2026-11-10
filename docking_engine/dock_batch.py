#!/usr/bin/env python3
"""
dock_batch.py — batch GPU docking against a receptor ensemble.
Ported directly from the working monolith dock_batch.py (CACHEBUST=60).
Adapted for split-container architecture.

Usage:
    python3 dock_batch.py \
        --receptor_dir /receptor \
        --library_dir /library \
        [--use_conformers] [--force]
"""

import argparse
import csv
import json
from reconstruct_hydrogens import (reconstruct_complex_hydrogens,
                                    build_ligand_h_template)
import shutil
import sqlite3
import subprocess
import complex_validity as cv
import sys
import time
from datetime import datetime
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# Library loading
# ══════════════════════════════════════════════════════════════════════════════

def load_library(library_dir: Path) -> list[dict]:
    db_path = library_dir / "library_state.db"
    if db_path.exists():
        try:
            con = sqlite3.connect(str(db_path), timeout=10)
            con.execute("PRAGMA journal_mode=WAL")
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM ligands WHERE status='ready' ORDER BY name"
            ).fetchall()
            con.close()
            ligands = []
            for row in rows:
                d = dict(row)
                d["bbox"]     = json.loads(d.get("bbox_json")     or "{}")
                d["elements"] = json.loads(d.get("elements_json") or "[]")
                ligands.append(d)
            return ligands
        except Exception as e:
            print(f"WARNING: Could not read library_state.db: {e}")
    json_path = library_dir / "library.json"
    if json_path.exists():
        lib = json.loads(json_path.read_text())
        return [v for v in lib.values() if v.get("status") == "ok"]
    return []


# ══════════════════════════════════════════════════════════════════════════════
# Ensemble loading — resolves container-internal paths to actual paths
# ══════════════════════════════════════════════════════════════════════════════

def load_ensemble(receptor_dir: Path) -> tuple[list[dict], dict]:
    ensemble_path = receptor_dir / "ensemble.json"
    if not ensemble_path.exists():
        raise FileNotFoundError(
            f"ensemble.json not found in {receptor_dir}\n"
            f"Run receptor prep first.")
    data = json.loads(ensemble_path.read_text())

    conformations = []
    for c in data.get("conformations", []):
        prep_dir = Path(c["prep_dir"])
        # Fallback: scan models dir if absolute path missing
        if not prep_dir.exists():
            mid = c.get("model_id")
            if mid:
                prep_dir = receptor_dir / "models" / f"model_{int(mid):03d}"
        if prep_dir.exists() and (prep_dir / "receptor_prep.json").exists():
            c = dict(c)
            c["prep_dir"] = str(prep_dir)
            conformations.append(c)

    # Last resort: scan models directory
    if not conformations:
        models_dir = receptor_dir / "models"
        if models_dir.exists():
            for model_dir in sorted(models_dir.iterdir()):
                if (model_dir / "receptor_prep.json").exists():
                    conformations.append({"prep_dir": str(model_dir),
                                          "model_id": model_dir.name})

    if not conformations:
        raise RuntimeError("No valid receptor conformations found in ensemble")

    return conformations, data.get("box", {})


def remap_prep_path(stored_path: str, prep_dir: Path) -> Path:
    """
    Resolve a path from receptor_prep.json.
    Paths are now written as absolute host paths — use directly.
    Falls back to relative-from-prep_dir if absolute path missing.
    """
    if not stored_path:
        return None
    p = Path(stored_path)
    if p.exists():
        return p
    # Fallback: derive from prep_dir
    parts = p.parts
    if "models" in parts:
        idx = parts.index("models")
        rel = parts[idx + 2:]
        if rel:
            candidate = prep_dir / Path(*rel)
            if candidate.exists():
                return candidate
    return prep_dir / p.name


# ══════════════════════════════════════════════════════════════════════════════
# Core docking (verbatim from working monolith)
# ══════════════════════════════════════════════════════════════════════════════

def dock_one(fld_file: Path, flex_pdbqt: Path | None,
             ligand_pdbqt: Path, name: str,
             out_dir: Path, nrun: int = 20) -> tuple[float | None, str]:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "autodock_gpu_128wi",
        "--ffile", str(fld_file),
        "--lfile", str(ligand_pdbqt),
        "--nrun",  str(nrun),
        "--resnam", name,
        "--lsmet", "ad",
        "--autostop", "1",
        "--derivtype", "H=HD",
    ]
    if flex_pdbqt and flex_pdbqt.exists():
        cmd += ["--flexres", str(flex_pdbqt)]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=3600, cwd=str(log_dir)
        )
        stdout = result.stdout.strip()
        failed = (result.returncode != 0
                  or "was not successful" in stdout.lower()
                  or "error in" in stdout.lower())

        if failed:
            if flex_pdbqt and "too many rotatable bonds" in stdout.lower():
                rigid_cmd = [c for c in cmd
                             if c != "--flexres" and c != str(flex_pdbqt)]
                r2 = subprocess.run(rigid_cmd, capture_output=True, text=True,
                                    timeout=3600, cwd=str(log_dir))
                stdout2 = r2.stdout.strip()
                if r2.returncode == 0 and "was not successful" not in stdout2.lower():
                    result = r2; stdout = stdout2; failed = False
                    print(f" [rigid fallback]", end="", flush=True)
                else:
                    return None, "too many torsions (rigid fallback also failed)"
            else:
                diag = "\n      ".join(
                    l for l in stdout.splitlines()
                    if l.strip() and not l.startswith("    ")
                ) if stdout else "(no stdout)"
                return None, f"exit {result.returncode}:\n      {diag}"

        # Find .dlg
        dlg_path = None
        for candidate in [log_dir / f"{name}_out.dlg", log_dir / f"{name}.dlg"]:
            if candidate.exists():
                dlg_path = candidate; break
        if dlg_path is None:
            hits = list(log_dir.glob(f"{name}*.dlg"))
            if hits: dlg_path = hits[0]

        if dlg_path is None:
            preview = stdout[-200:] if stdout else "(empty)"
            return None, f"no .dlg produced. stdout: {preview}"

        score = parse_dlg_score(dlg_path)
        if score is None:
            return None, "could not parse score from .dlg"
        return score, "OK"

    except subprocess.TimeoutExpired:
        return None, "docking timeout"
    except FileNotFoundError:
        return None, "autodock_gpu_128wi not found"
    except Exception as e:
        return None, str(e)


def parse_dlg_score(dlg_path: Path) -> float | None:
    best = None
    with open(dlg_path) as f:
        for line in f:
            if "Estimated Free Energy of Binding" in line and "=" in line:
                try:
                    val = float(line.split("=")[1].split()[0])
                    if best is None or val < best:
                        best = val
                except (ValueError, IndexError):
                    pass
    return round(best, 3) if best is not None else None


# ══════════════════════════════════════════════════════════════════════════════
# Complex builder (verbatim from working monolith)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_flex_pdbqt_atoms(flex_pdbqt: Path) -> list[dict]:
    atoms, current_res = [], None
    with open(flex_pdbqt) as f:
        for line in f:
            if line.startswith("BEGIN_RES"):
                parts = line.split()
                if len(parts) >= 4:
                    current_res = (parts[2], parts[1], parts[3])
            elif line.startswith("END_RES"):
                current_res = None
            elif line.startswith("ATOM") and current_res is not None:
                atoms.append({
                    "chain":   current_res[0],
                    "resname": current_res[1],
                    "resseq":  current_res[2],
                    "aname":   line[12:16].strip(),
                })
    return atoms


def build_complex(fixed_pdb: Path, dlg_path: Path,
                  out_pdb: Path, name: str,
                  n_lig_atoms: int = 0,
                  flex_pdbqt: Path | None = None,
                  h_template: dict | None = None,
                  core_mask: dict | None = None) -> dict:
    runs: dict = {}
    current_run = current_e = None
    current_lines = []
    in_run = False

    with open(dlg_path) as f:
        for line in f:
            if "DOCKED: MODEL" in line:
                try:    current_run = int(line.split()[-1])
                except: current_run = len(runs) + 1
                current_e = None; current_lines = []; in_run = True
            elif "DOCKED: ENDMDL" in line:
                if current_run is not None and current_e is not None:
                    runs[current_run] = (current_e, list(current_lines))
                in_run = False
            elif in_run:
                stripped = line[8:] if line.startswith("DOCKED: ") else line
                if "Estimated Free Energy of Binding" in stripped and "=" in stripped:
                    try: current_e = float(stripped.split("=")[1].split()[0])
                    except: pass
                if stripped.startswith(("ATOM","HETATM")):
                    current_lines.append(stripped)

    if not runs:
        return {"ok": False, "status": "no_complex", "energy": None,
                "rank": -1, "n_runs": 0, "reranked": False}

    # Receptor template and flexible-residue metadata, read once.
    with open(fixed_pdb) as rec:
        fixed_lines = rec.readlines()
    flex_meta = (_parse_flex_pdbqt_atoms(flex_pdbqt)
                 if flex_pdbqt and flex_pdbqt.exists() else [])

    def _is_h(line):
        """True if this PDB/PDBQT line is a hydrogen atom."""
        aname = line[12:16].strip() if len(line) > 16 else ""
        elem  = line[76:78].strip() if len(line) > 76 else ""
        ad4   = line[77:79].strip() if len(line) > 77 else ""
        if elem in ("H","D") or ad4 in ("H","HD"):
            return True
        if aname.startswith("H") and len(aname) <= 4:
            return True
        return False

    def _coord_updates(run_lines):
        """Flex-residue coordinate overrides for one run, keyed as the receptor
        lines are keyed, so each run's own docked sidechains are in place before
        the validity check — what is validated is what is written."""
        if not (n_lig_atoms > 0 and len(run_lines) > n_lig_atoms and flex_meta):
            return {}
        upd = {}
        for i, dl in enumerate(run_lines[n_lig_atoms:]):
            if i >= len(flex_meta): break
            m = flex_meta[i]
            try:
                upd[(m["chain"], m["resname"], m["resseq"], m["aname"])] = (
                    dl[30:38], dl[38:46], dl[46:54])
            except IndexError:
                continue
        return upd

    def _assemble(run_lines):
        """Assemble heavy-atom complex lines for one run: receptor with this
        run's flex applied, TER, ligand as HETATM. Hydrogens stripped."""
        lig_lines = (run_lines[:n_lig_atoms]
                     if n_lig_atoms > 0 and len(run_lines) > n_lig_atoms
                     else run_lines)
        lig_out = [f"HETATM{l[6:17]}{name[:3].upper()} A 999{l[26:]}"
                   for l in lig_lines if not _is_h(l)]
        upd = _coord_updates(run_lines)
        out = []
        for line in fixed_lines:
            if line.startswith(("ATOM","HETATM")) and upd:
                try:
                    key = (line[21].strip(), line[17:20].strip(),
                           line[22:26].strip(), line[12:16].strip())
                    if key in upd:
                        x, y, z = upd[key]
                        line = line[:30] + x + y + z + line[54:]
                except IndexError:
                    pass
            if line.startswith(("ATOM","HETATM","TER")):
                if line.startswith(("ATOM","HETATM")) and _is_h(line):
                    continue
                out.append(line if line.endswith("\n") else line + "\n")
        out.append("TER\n")
        out.extend(l if l.endswith("\n") else l + "\n" for l in lig_out)
        out.append("END\n")
        return out

    def _verdict(lines):
        """complex_validity verdict for assembled lines. The ligand is the only
        HETATM, so records split by record type — one clash/pierce metric for
        the whole pipeline."""
        rec_r, lig_r = [], []
        for ln in lines:
            if ln.startswith("HETATM"):
                a = cv._parse_atom(ln)
                if a: lig_r.append(a)
            elif ln.startswith("ATOM"):
                a = cv._parse_atom(ln)
                if a: rec_r.append(a)
        return cv.validate(lig_r, rec_r, clash_max=0, core_mask=core_mask)

    # Walk the GA runs best-energy first and keep the first physically valid
    # one — no clashes over the limit AND no bond threaded through a ring. A
    # good score on a ring-pierced pose is never accepted; only if no run is
    # valid do we fall back to the lowest-energy pose and record why it failed.
    ranked = sorted(runs.values(), key=lambda x: x[0])
    n_runs = len(ranked)
    chosen = fallback = None
    for rank, (energy, run_lines) in enumerate(ranked):
        lines = _assemble(run_lines)
        v = _verdict(lines)
        if fallback is None:
            fallback = (energy, lines, v, rank)      # lowest energy
        if v["valid"]:
            chosen = (energy, lines, v, rank)
            break
    if chosen is None:
        chosen = fallback

    energy, lines, v, rank = chosen
    with open(out_pdb, "w") as f:
        f.writelines(lines)
    try:
        reconstruct_complex_hydrogens(out_pdb, ligand_template=h_template)
    except Exception:
        pass  # Non-fatal — complex kept without hydrogens

    if v["valid"]:
        status = "OK" if rank == 0 else f"OK(valid pose {rank+1}/{n_runs})"
    else:
        parts = []
        if v["clashes"] > 0:
            parts.append(f"CLASH({v['clashes']} pairs)")
        if v["piercings"]:
            rings = ",".join(sorted({p["ring"] for p in v["piercings"]}))
            parts.append(f"PIERCE({len(v['piercings'])}:{rings})")
        if not v.get("core_ok", True):
            parts.append(f"SURFACE({v['core_fraction']:.0%} in core)")
        status = ";".join(parts) if parts else "OK"
    return {"ok": True, "status": status, "energy": energy,
            "rank": rank, "n_runs": n_runs,
            "reranked": bool(v["valid"] and rank > 0)}


# validate_complex() was removed: complex validity (clashes AND bonds threaded
# through rings) is now computed inside build_complex() via complex_validity,
# the single clash/pierce metric for the whole pipeline.


# ══════════════════════════════════════════════════════════════════════════════
# Results management
# ══════════════════════════════════════════════════════════════════════════════

def load_existing_results(results_csv: Path) -> dict:
    if not results_csv.exists(): return {}
    out = {}
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            out[row["name"]] = row
    return out


def archive_result(results_dir: Path, ligand_name: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    complex_path = results_dir/"complexes"/f"complex_{ligand_name}.pdb"
    if complex_path.exists():
        arch = results_dir/"complexes"/"archive"
        arch.mkdir(exist_ok=True)
        shutil.copy2(complex_path, arch/f"complex_{ligand_name}_{ts}.pdb")
    csv_path = results_dir/"results.csv"
    if csv_path.exists():
        arch = results_dir/"results_archive"
        arch.mkdir(exist_ok=True)
        shutil.copy2(csv_path, arch/f"results_{ts}.csv")


def write_results_csv(results_dir: Path, rows: list[dict]):
    if not rows: return
    fields = ["name","formula","smiles","score","status","validation",
              "complex","n_receptor_models","n_ligand_conformers","docked_at"]
    sorted_rows = sorted(rows,
        key=lambda r: float(r["score"]) if r.get("score") not in ("",None) else 999)
    with open(results_dir/"results.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(sorted_rows)


def _scale_nrun(n_rot: int, base: int) -> int:
    if n_rot > 35: return max(base, 100)
    if n_rot > 20: return max(base, 50)
    return base


def compute_delta(ligands, conformations, existing, force) -> list:
    if force: return ligands
    to_dock = []
    for lig in ligands:
        name = lig["name"]
        if name not in existing:
            to_dock.append(lig); continue
        prev = existing[name]
        if int(lig.get("n_conformers",0)) > int(prev.get("n_ligand_conformers",0)):
            to_dock.append(lig); continue
        if len(conformations) > int(prev.get("n_receptor_models",0)):
            to_dock.append(lig); continue
    return to_dock


# ══════════════════════════════════════════════════════════════════════════════
# Main docking loop
# ══════════════════════════════════════════════════════════════════════════════

def dock_ligand(entry: dict, conformations: list[dict],
                results_dir: Path, nrun_base: int,
                use_conformers: bool) -> dict:
    name     = entry["name"]
    n_rot    = entry.get("n_rot_bonds", 0)
    pdbqt    = Path(entry.get("pdbqt_path") or entry.get("pdbqt",""))
    flex_fb  = bool(entry.get("use_flex_fallback", False))
    log_dir  = results_dir / "logs"
    cplx_dir = results_dir / "complexes"
    cplx_dir.mkdir(parents=True, exist_ok=True)

    if use_conformers and entry.get("conf_dir") and not flex_fb:
        conf_dir    = Path(entry["conf_dir"])
        conf_pdbqts = sorted(conf_dir.glob("conf_*.pdbqt"))
        if conf_pdbqts:
            dock_pdbqts    = conf_pdbqts
            effective_nrun = 1
            mode_tag       = f"conformers({len(conf_pdbqts)})"
        else:
            dock_pdbqts    = [pdbqt]
            effective_nrun = _scale_nrun(n_rot, nrun_base)
            mode_tag       = "flex"
    else:
        dock_pdbqts    = [pdbqt]
        effective_nrun = _scale_nrun(n_rot, nrun_base)
        mode_tag       = "flex-fallback" if flex_fb else "flex"

    n_lig  = len(dock_pdbqts)
    n_rec  = len(conformations)
    total  = n_rec * n_lig * effective_nrun

    # Whether flex residues will be supplied is now uniform across modes,
    # but surface it in the log so a maps/flex mismatch is visible at a glance.
    flex_note = ""
    try:
        _m = json.loads(
            (Path(conformations[0]["prep_dir"]) / "receptor_prep.json").read_text())
        flex_note = "  flex=on" if _m.get("flex_enabled") else "  flex=none"
    except Exception:
        pass

    print(f"[{mode_tag}  R={n_rec}  L={n_lig}  "
          f"nrun={effective_nrun}  total={total}{flex_note}]", end="  ", flush=True)

    best_score = None; best_status = "no conformations tried"
    best_dlg   = None; best_fixed  = None; best_flex = None
    best_core_mask = None

    for r_idx, conf in enumerate(conformations):
        prep_dir = Path(conf["prep_dir"])
        r_meta   = json.loads((prep_dir/"receptor_prep.json").read_text())

        # Remap container-internal paths to actual paths
        r_fld   = remap_prep_path(r_meta.get("fld_file"),    prep_dir)
        r_fixed = remap_prep_path(r_meta.get("fixed_pdb"),   prep_dir)
        r_rigid = remap_prep_path(r_meta.get("rigid_pdbqt"), prep_dir)
        r_flex_stored = r_meta.get("flex_pdbqt")
        r_flex  = remap_prep_path(r_flex_stored, prep_dir) if r_flex_stored else None

        if not r_fld or not r_fld.exists():
            print(f"\n  [SKIP R{r_idx+1}] fld not found: {r_fld}", flush=True)
            continue
        if not r_fixed or not r_fixed.exists():
            print(f"\n  [SKIP R{r_idx+1}] fixed_pdb not found: {r_fixed}", flush=True)
            continue

        # Flex residues must be supplied whenever the maps were built from
        # the split receptor. split_flex_receptor() removes the selected
        # sidechain atoms before generate_maps() runs, so withholding
        # --flexres docks the ligand into the void those sidechains left;
        # build_complex() then restores them from receptor_fixed.pdb,
        # directly through the pose. That mismatch produced ~121 clashing
        # pairs per ligand in conformer mode against ~0 in flex mode.
        lig_flex = r_flex

        if r_meta.get("flex_enabled") and not (r_flex and r_flex.exists()):
            print(f"\n  [WARN R{r_idx+1}] maps were built from the split "
                  f"receptor but no flex PDBQT is available — poses will "
                  f"clash with the missing sidechains", flush=True)

        for c_idx, lig_pdbqt in enumerate(dock_pdbqts):
            if not lig_pdbqt.exists():
                print(f"\n  [SKIP] pdbqt not found: {lig_pdbqt}", flush=True)
                continue
            run_name = f"{name}_r{r_idx}_c{c_idx}"
            sc, st   = dock_one(r_fld, lig_flex, lig_pdbqt,
                                 run_name, results_dir, effective_nrun)
            if sc is not None and (best_score is None or sc < best_score):
                best_score  = sc
                best_status = st
                best_fixed  = r_fixed
                best_flex   = r_flex
                best_core_mask = r_meta.get("core_mask")
                for cand in [log_dir/f"{run_name}_out.dlg",
                              log_dir/f"{run_name}.dlg"]:
                    if cand.exists():
                        best_dlg = cand; break
                if best_dlg is None:
                    hits = list(log_dir.glob(f"{run_name}*.dlg"))
                    if hits: best_dlg = hits[0]

        if n_rec > 1 and best_score is not None:
            print(f" [R{r_idx+1}:{best_score:.2f}]", end="", flush=True)

    # Count ligand atoms for flex/lig split in complex builder
    n_lig_atoms = 0
    try:
        with open(pdbqt) as f:
            n_lig_atoms = sum(1 for l in f if l.startswith(("ATOM","HETATM")))
    except Exception: pass

    # Load ligand hydrogen template for reconstruction
    h_template = None
    try:
        tmpl_path = pdbqt.parent / "hydrogen_template.json"
        if tmpl_path.exists():
            h_template = json.loads(tmpl_path.read_text())
    except Exception:
        pass

    # Build complex
    validation   = "no_complex"
    complex_path = None

    if best_dlg and best_fixed and best_score is not None:
        final_dlg = log_dir / f"{name}_best.dlg"
        if best_dlg != final_dlg:
            shutil.copy2(best_dlg, final_dlg)
        out_pdb = cplx_dir / f"complex_{name}.pdb"
        res = build_complex(best_fixed, final_dlg, out_pdb, name,
                            n_lig_atoms, best_flex, h_template,
                            core_mask=best_core_mask)
        if res["ok"]:
            complex_path = str(out_pdb)
            validation   = res["status"]
            if res.get("energy") is not None:
                best_score = res["energy"]   # score of the pose actually kept

    return {
        "name":                name,
        "formula":             entry.get("formula",""),
        "smiles":              entry.get("smiles",""),
        "score":               f"{best_score:.3f}" if best_score is not None else "",
        "status":              best_status,
        "validation":          validation,
        "complex":             complex_path or "",
        "n_receptor_models":   n_rec,
        "n_ligand_conformers": n_lig,
        "docked_at":           datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _select_screening_model(conformations: list[dict],
                             box: dict) -> int:
    """
    Select the conformation with the fewest Cα atoms inside the search box.
    Returns model_id of the recommended starting model.
    """
    cx, cy, cz = box.get("cx",0), box.get("cy",0), box.get("cz",0)
    sx, sy, sz = box.get("sx",25), box.get("sy",25), box.get("sz",25)

    best_id    = conformations[0].get("model_id", 1)
    best_count = None

    for conf in conformations:
        fixed_pdb = Path(conf["prep_dir"]) / "receptor_fixed.pdb"
        if not fixed_pdb.exists():
            continue
        count = 0
        with open(fixed_pdb) as f:
            for line in f:
                if not line.startswith("ATOM"): continue
                if line[12:16].strip() != "CA": continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    if (abs(x-cx) <= sx/2 and
                        abs(y-cy) <= sy/2 and
                        abs(z-cz) <= sz/2):
                        count += 1
                except (ValueError, IndexError):
                    continue
        if best_count is None or count < best_count:
            best_count = count
            best_id    = conf.get("model_id", 1)

    return best_id


def main():
    p = argparse.ArgumentParser(description="Batch GPU docking.")
    p.add_argument("--job",            help="Job JSON file (preferred)")
    p.add_argument("--receptor_dir",   default=None)
    p.add_argument("--library_dir",    default=None)
    p.add_argument("--nrun",           type=int, default=20)
    p.add_argument("--use_conformers", action="store_true")
    p.add_argument("--force",          action="store_true")
    p.add_argument("--max_failures",   type=int, default=None)
    p.add_argument("--compounds",      nargs="+", metavar="NAME")
    p.add_argument("--min_atoms",      type=int, default=None)
    p.add_argument("--max_atoms",      type=int, default=None)
    p.add_argument("--min_rot",        type=int, default=None)
    p.add_argument("--max_rot",        type=int, default=None)
    args = p.parse_args()

    # Overlay job file values
    if args.job:
        with open(args.job) as f:
            job_data = json.load(f)
        opts = job_data.get("options", {})
        if job_data.get("receptor_dir"):    args.receptor_dir   = job_data["receptor_dir"]
        if job_data.get("library_dir"):     args.library_dir    = job_data["library_dir"]
        if opts.get("nrun") is not None:    args.nrun           = int(opts["nrun"])
        if opts.get("use_conformers"):      args.use_conformers = bool(opts["use_conformers"])
        if opts.get("force"):               args.force          = bool(opts["force"])
        if opts.get("max_failures"):        args.max_failures   = int(opts["max_failures"])
        if opts.get("compounds"):           args.compounds      = opts["compounds"]
        if opts.get("max_atoms"):           args.max_atoms      = int(opts["max_atoms"])
        if opts.get("max_rot"):             args.max_rot        = int(opts["max_rot"])

    if not args.receptor_dir:
        print("ERROR: --receptor_dir or --job required", file=sys.stderr); sys.exit(1)
    if not args.library_dir:
        print("ERROR: --library_dir or --job required", file=sys.stderr); sys.exit(1)

    receptor_dir = Path(args.receptor_dir)
    library_dir  = Path(args.library_dir)

    print(f"\n=== DOCKING ENGINE ===")
    print(f"  Receptor: {receptor_dir.name}")

    conformations, box = load_ensemble(receptor_dir)
    print(f"  Models:   {len(conformations)} receptor conformation(s)")

    # Select best starting model for screening:
    # prefer the model with the fewest Cα atoms inside the search box
    # (less steric occlusion → better chance of finding early hits)
    if len(conformations) > 1 and box:
        best_start = _select_screening_model(conformations, box)
        if best_start != conformations[0].get("model_id"):
            # Reorder so best_start is first
            idx = next((i for i, c in enumerate(conformations)
                        if c.get("model_id") == best_start), 0)
            conformations = ([conformations[idx]] +
                             conformations[:idx] +
                             conformations[idx+1:])
            print(f"  Screen:   starting with model {best_start} "
                  f"(fewest atoms in search box)")

    ligands = load_library(library_dir)
    if not ligands:
        print("ERROR: No ready ligands in library", file=sys.stderr); sys.exit(1)

    if args.compounds:
        keep = set(args.compounds)
        ligands = [l for l in ligands if l["name"] in keep]
    if args.min_atoms: ligands = [l for l in ligands if l.get("n_heavy_atoms",0) >= args.min_atoms]
    if args.max_atoms: ligands = [l for l in ligands if l.get("n_heavy_atoms",0) <= args.max_atoms]
    if args.min_rot:   ligands = [l for l in ligands if l.get("n_rot_bonds",0)   >= args.min_rot]
    if args.max_rot:   ligands = [l for l in ligands if l.get("n_rot_bonds",0)   <= args.max_rot]
    if not ligands:
        print("ERROR: No ligands after filtering", file=sys.stderr); sys.exit(1)

    print(f"  Ligands:  {len(ligands)} in library after filters")

    results_dir = receptor_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir/"complexes").mkdir(exist_ok=True)
    (results_dir/"logs").mkdir(exist_ok=True)

    existing = load_existing_results(results_dir/"results.csv")
    to_dock  = compute_delta(ligands, conformations, existing, args.force)
    print(f"  To dock:  {len(to_dock)} "
          f"({len(ligands)-len(to_dock)} skipped — already docked, no changes)")
    if args.force: print(f"  Mode:     --force (re-docking all)")
    print()

    if not to_dock:
        print("Nothing to dock. Use --force to re-dock existing results.")
        sys.exit(0)

    all_results    = dict(existing)
    total          = len(to_dock)
    n_w            = len(str(total))
    t_start        = time.time()
    consecutive_fail = 0

    for i, entry in enumerate(to_dock, 1):
        name  = entry["name"]
        t_lig = time.time()
        print(f"  [{i:{n_w}}/{total}] {name:<30}", end="  ", flush=True)

        pdbqt = Path(entry.get("pdbqt_path") or entry.get("pdbqt",""))
        if not pdbqt.exists():
            print(f"FAILED (PDBQT missing)")
            all_results[name] = {
                "name":name,"formula":entry.get("formula",""),
                "smiles":entry.get("smiles",""),
                "score":"","status":"pdbqt missing","validation":"no_complex",
                "complex":"","n_receptor_models":len(conformations),
                "n_ligand_conformers":0,"docked_at":datetime.now().isoformat()}
            consecutive_fail += 1
            if args.max_failures and consecutive_fail >= args.max_failures:
                print(f"\n  Early termination: {consecutive_fail} consecutive failures")
                break
            continue

        if name in existing and args.force:
            archive_result(results_dir, name)

        row     = dock_ligand(entry, conformations, results_dir,
                              args.nrun, args.use_conformers)
        elapsed = time.time() - t_lig
        score_str = (f"score {float(row['score']):>8.3f} kcal/mol"
                     if row["score"] else "FAILED")
        print(f"{score_str}  [{row['validation']}]  ({elapsed:.1f}s)")

        # Keep better score if re-docking
        if name in existing and row["score"] and existing[name].get("score"):
            try:
                if float(row["score"]) > float(existing[name]["score"]):
                    print(f"        NOTE: previous score {existing[name]['score']} was better")
                    row = existing[name]
            except (ValueError, TypeError): pass

        all_results[name] = row
        consecutive_fail = 0 if row["score"] else consecutive_fail + 1
        if args.max_failures and consecutive_fail >= args.max_failures:
            print(f"\n  Early termination: {consecutive_fail} consecutive failures")
            break

    write_results_csv(results_dir, list(all_results.values()))

    t_total = time.time() - t_start
    n_ok    = sum(1 for r in all_results.values() if r.get("score"))
    n_clash = sum(1 for r in all_results.values()
                  if str(r.get("validation","")).startswith("CLASH"))
    print(f"\n=== COMPLETE ===")
    print(f"  Docked:    {len(to_dock)} ligands in {t_total:.0f}s")
    print(f"  Scored:    {n_ok}")
    print(f"  Clashes:   {n_clash} complex(es) flagged")
    print(f"  Results:   {results_dir/'results.csv'}")
    print(f"  Complexes: {results_dir/'complexes'}")


if __name__ == "__main__":
    main()
