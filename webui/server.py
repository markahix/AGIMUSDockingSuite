#!/usr/bin/env python3
"""
server.py — AutoDock Pipeline Web UI Orchestrator
Deposits files, reads state, submits job files.
Containers mirror host paths exactly — no path translation needed.
"""

import csv
import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

from flask import (Flask, jsonify, redirect, render_template,
                   request, send_file, url_for)
from flask_socketio import SocketIO, emit

from job_writer import (submit_dock_job, submit_prep_ligands_job,
                        submit_prep_receptor_job)
from state_reader import (get_all_receptors, get_inbox_contents,
                           get_job_queue, get_queue_view, get_library_ligands,
                           get_library_summary, get_receptor_models,
                           get_receptor_results)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", uuid.uuid4().hex)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
sio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# ── Directory paths (host absolute paths passed via environment) ───────────────
LIGAND_DIR    = Path(os.environ.get("LIGAND_DIR",    "/ligands"))
LIBRARY_DIR   = Path(os.environ.get("LIBRARY_DIR",   "/library"))
RECEPTORS_DIR = Path(os.environ.get("RECEPTORS_DIR", "/receptors"))
JOBS_DIR      = Path(os.environ.get("JOBS_DIR",      "/jobs"))


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use as a directory/filename."""
    name = name.strip()
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name[:64] or "receptor"


def _receptor_dir(name: str) -> Path:
    return RECEPTORS_DIR / name


# ══════════════════════════════════════════════════════════════════════════════
# Box centre helpers
# ══════════════════════════════════════════════════════════════════════════════

_STANDARD_AA = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "HSD","HSE","HSP","HIE","HID","HIP","CYX","CYM","GLH","ASH",
    "LYN","ARN","SEC","PYL","MSE",
    "DA","DC","DG","DT","DU","A","C","G","T","U",
    "HOH","WAT","SOL","NA","CL","MG","ZN","CA","K","FE","MN",
}


def _centroid_from_residues(pdb_path: Path, mask: str) -> tuple:
    accepted_nums, accepted_names = set(), set()
    for part in mask.replace(" ", "").split(","):
        part = part.strip(":")
        if "-" in part:
            try:
                lo, hi = part.split("-")
                accepted_nums.update(range(int(lo), int(hi)+1))
            except ValueError:
                accepted_names.add(part.upper())
        elif part.isdigit():
            accepted_nums.add(int(part))
        else:
            accepted_names.add(part.upper())
    xs, ys, zs = [], [], []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"): continue
            if line[12:16].strip() != "CA": continue
            try:
                resname = line[17:20].strip().upper()
                resseq  = int(line[22:26])
                if resseq not in accepted_nums and resname not in accepted_names:
                    continue
                xs.append(float(line[30:38]))
                ys.append(float(line[38:46]))
                zs.append(float(line[46:54]))
            except (ValueError, IndexError):
                continue
    if not xs:
        raise ValueError(f"No Cα atoms found matching: {mask}")
    return sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)


def _centroid_from_hetatm(pdb_path: Path, resname: str | None) -> tuple:
    xs, ys, zs = [], [], []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")): continue
            rname = line[17:20].strip().upper()
            if resname:
                if rname != resname.upper(): continue
            else:
                if line.startswith("ATOM  ") and rname in _STANDARD_AA: continue
                if rname in ("HOH","WAT","SOL","NA","CL","MG","ZN","CA","K","FE","MN"):
                    continue
            try:
                xs.append(float(line[30:38]))
                ys.append(float(line[38:46]))
                zs.append(float(line[46:54]))
            except (ValueError, IndexError):
                continue
    if not xs:
        raise ValueError(
            f"No ligand atoms found" +
            (f" with resname '{resname}'" if resname else
             " — try specifying the residue name explicitly"))
    return sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs)


def _detect_active_sites_from_pdb(pdb_path: Path,
                                   resnames_str: str,
                                   padding: float = 10.0) -> list[dict]:
    """
    Parse comma-separated resnames, detect all instances in PDB,
    cluster overlapping boxes, and return active site list.

    An empty resnames_str means auto-detect: every residue that is not a
    standard polymer residue, solvent, or common ion is treated as a
    candidate ligand. Both ATOM and HETATM records are scanned.
    """
    from active_sites import detect_active_sites as _das
    resnames = [r.strip().upper()
                for r in resnames_str.replace(",", " ").split()
                if r.strip()]
    return _das(pdb_path, resnames or None, padding)


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def dashboard():
    return render_template("dashboard.html",
                           summary=get_library_summary(LIBRARY_DIR),
                           receptors=get_all_receptors(RECEPTORS_DIR),
                           jobs=get_job_queue(JOBS_DIR),
                           inbox=get_inbox_contents(RECEPTORS_DIR))


@app.route("/api/status")
def api_status():
    return jsonify({
        "library":   get_library_summary(LIBRARY_DIR),
        "receptors": len(get_all_receptors(RECEPTORS_DIR)),
        "jobs":      get_job_queue(JOBS_DIR),
    })


# ══════════════════════════════════════════════════════════════════════════════
# Library
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/queue")
def queue_page():
    return render_template("queue.html", queue=get_queue_view(JOBS_DIR))


@app.route("/api/queue")
def api_queue():
    return jsonify(get_queue_view(JOBS_DIR))


@app.route("/library")
def library_page():
    return render_template("library.html",
                           ligands=get_library_ligands(LIBRARY_DIR),
                           summary=get_library_summary(LIBRARY_DIR))


@app.route("/library/upload", methods=["POST"])
def library_upload():
    files = request.files.getlist("pdbs")
    n_saved, errors = 0, []
    LIGAND_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        if not f.filename: continue
        name = Path(f.filename).name
        if not name.lower().endswith(".pdb"):
            errors.append(f"{name}: not a PDB file"); continue
        f.save(LIGAND_DIR / name)
        n_saved += 1
    if n_saved:
        return jsonify({"saved": n_saved, "errors": errors})
    return jsonify({"error": "No files saved", "errors": errors}), 400


@app.route("/library/prep", methods=["POST"])
def library_prep():
    data = request.get_json() or {}
    try:
        job_id = submit_prep_ligands_job(
            JOBS_DIR,
            library_dir=str(LIBRARY_DIR),
            ligand_dir=str(LIGAND_DIR),
            options={
                "refresh_all": data.get("refresh_all", False),
                "conformers":  data.get("conformers"),
            }
        )
        return jsonify({"job_id": job_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/library/ligands")
def api_library_ligands():
    return jsonify(get_library_ligands(LIBRARY_DIR))


# ══════════════════════════════════════════════════════════════════════════════
# Receptors
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/receptors")
def receptors_page():
    return render_template("receptors.html",
                           receptors=get_all_receptors(RECEPTORS_DIR),
                           inbox=get_inbox_contents(RECEPTORS_DIR))


@app.route("/receptors/upload", methods=["POST"])
def receptor_upload():
    """
    Upload a receptor PDB. The receptor name (from the form) is used as
    the filename in the inbox — this becomes the receptor directory name.
    """
    f = request.files.get("pdb")
    if not f or not f.filename:
        return jsonify({"error": "No file provided"}), 400
    if not f.filename.lower().endswith(".pdb"):
        return jsonify({"error": "Not a PDB file"}), 400

    # Use provided name or fall back to filename stem
    raw_name = request.form.get("receptor_name", "").strip()
    if not raw_name:
        raw_name = Path(f.filename).stem
    safe_name = _sanitize_name(raw_name)
    filename  = safe_name + ".pdb"

    inbox = RECEPTORS_DIR / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    f.save(inbox / filename)
    return jsonify({"filename": filename, "receptor_name": safe_name})


@app.route("/receptors/prep", methods=["POST"])
def receptor_prep():
    data         = request.get_json() or {}
    pdb_filename = data.get("pdb_filename")
    box          = data.get("box", {})

    if not pdb_filename:
        return jsonify({"error": "pdb_filename required"}), 400

    safe_filename = _sanitize_name(Path(pdb_filename).stem) + ".pdb"
    inbox_path    = RECEPTORS_DIR / "inbox" / safe_filename
    if not inbox_path.exists():
        return jsonify({"error": f"PDB not in inbox: {pdb_filename}"}), 404

    # Resolve box centre
    mode = box.get("mode", "coords")
    cavity_mode = bool(box.get("cavity"))
    cx = cy = cz = None
    sx = sy = sz = None
    if mode == "coords":
        cx, cy, cz = box.get("cx"), box.get("cy"), box.get("cz")
        if None in (cx, cy, cz):
            return jsonify({"error": "cx, cy, cz required"}), 400
        # Multi-site detection may pass explicit sizes — use them directly
        if box.get("size_x") and box.get("size_y") and box.get("size_z"):
            sx = float(box["size_x"])
            sy = float(box["size_y"])
            sz = float(box["size_z"])
    elif mode == "residues":
        mask = box.get("residue_mask", "").strip()
        if not mask:
            return jsonify({"error": "residue_mask required"}), 400
        try:
            cx, cy, cz = _centroid_from_residues(inbox_path, mask)
        except Exception as e:
            return jsonify({"error": f"Residue centroid: {e}"}), 400
    elif mode == "ligand":
        resname = box.get("hetatm_resname", "").strip() or None
        try:
            cx, cy, cz = _centroid_from_hetatm(inbox_path, resname)
        except Exception as e:
            return jsonify({"error": f"HETATM centroid: {e}"}), 400
    else:
        return jsonify({"error": f"Unknown box mode: {mode}"}), 400

    # Metadata
    meta = {
        "name":     data.get("receptor_name", Path(safe_filename).stem),
        "variant":  data.get("variant",  ""),
        "gene_id":  data.get("gene_id",  ""),
        "gene_db":  data.get("gene_db",  ""),
        "notes":    data.get("notes",    ""),
    }

    # Compute box size from library — webui has direct DB access,
    # containers do not (WAL lock issue). Pass explicit sizes in job file.
    # Multi-site detection may have pre-computed sx/sy/sz — use them if present.
    # Cavity mode skips this: the box is measured around the seed at prep.
    if not cavity_mode and not (sx and sy and sz):
        sx = sy = sz = None
        try:
            import sqlite3 as _sq
            _db = LIBRARY_DIR / "library_state.db"
            if _db.exists():
                _con = _sq.connect(str(_db), timeout=5)
                _rows = _con.execute(
                    "SELECT bbox_json FROM ligands WHERE status='ready'"
                    " AND bbox_json IS NOT NULL"
                ).fetchall()
                _con.close()
                if _rows:
                    _bboxes = [json.loads(r[0]) for r in _rows]
                    _pad = float(box.get("padding", 10.0))
                    sx = round(max(b["dx"] for b in _bboxes) + 2*_pad, 1)
                    sy = round(max(b["dy"] for b in _bboxes) + 2*_pad, 1)
                    sz = round(max(b["dz"] for b in _bboxes) + 2*_pad, 1)
        except Exception:
            pass

    # HETATM removal options
    remove_raw = data.get("remove_resnames", "").strip()
    if remove_raw == "":
        remove_resnames = None          # default: remove all HETATM
    elif remove_raw.upper() == "NONE":
        remove_resnames = []            # keep all HETATM
    else:
        remove_resnames = [r.strip().upper()
                           for r in re.split(r"[,\s]+", remove_raw)
                           if r.strip()]
    keep_water = bool(data.get("keep_water", False))

    try:
        job_id = submit_prep_receptor_job(
            JOBS_DIR,
            pdb_path=str(inbox_path),
            receptors_dir=str(RECEPTORS_DIR),
            library_dir=str(LIBRARY_DIR),
            center_x=float(cx),
            center_y=float(cy),
            center_z=float(cz),
            padding=float(box.get("padding", 10.0)),
            rmsd_threshold=float(box.get("rmsd_threshold", 0.5)),
            size_x=sx,
            size_y=sy,
            size_z=sz,
            remove_resnames=remove_resnames,
            keep_water=keep_water,
            box_mode="cavity" if cavity_mode else None,
            apron_reach=float(box.get("apron_reach", 8.0)),
            metadata=meta,
        )
        return jsonify({"job_id": job_id,
                        "center": {"x": cx, "y": cy, "z": cz},
                        "receptor_name": meta["name"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/receptors/<name>")
def receptor_detail(name: str):
    rec_dir = _receptor_dir(name)
    if not rec_dir.exists():
        return "Receptor not found", 404
    return render_template("receptor_detail.html",
                           name=name,
                           models=get_receptor_models(rec_dir),
                           results=get_receptor_results(rec_dir))


@app.route("/receptors/<name>/delete", methods=["POST"])
def receptor_delete(name: str):
    """Delete a receptor directory and all its contents."""
    # Sanitize name — must not contain path traversal
    safe = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    if safe != name or not safe:
        return jsonify({"error": "Invalid receptor name"}), 400
    rec_dir = _receptor_dir(safe)
    if not rec_dir.exists():
        return jsonify({"error": "Receptor not found"}), 404
    try:
        shutil.rmtree(rec_dir)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/receptors/detect_sites", methods=["POST"])
def detect_sites():
    """
    Given a PDB filename in the inbox and an optional comma-separated list
    of ligand residue names, return detected active sites with boxes.

    Both ATOM and HETATM records are scanned. A blank resnames field means
    auto-detect every non-polymer, non-solvent residue.
    """
    data       = request.get_json() or {}
    pdb_fname  = data.get("pdb_filename", "")
    resnames   = data.get("resnames", "")
    padding    = float(data.get("padding", 10.0))

    if not pdb_fname:
        return jsonify({"error": "pdb_filename required"}), 400

    safe_fname = _sanitize_name(Path(pdb_fname).stem) + ".pdb"
    inbox_path = RECEPTORS_DIR / "inbox" / safe_fname
    if not inbox_path.exists():
        return jsonify({"error": f"PDB not in inbox: {pdb_fname}"}), 404

    try:
        sites = _detect_active_sites_from_pdb(inbox_path, resnames, padding)
        if not sites:
            # Report what candidate residues the file actually contains so
            # the user can correct the resname rather than guess.
            from active_sites import parse_ligand_instances
            found = sorted(parse_ligand_instances(inbox_path, None).keys())
            return jsonify({
                "sites": [],
                "candidates": found,
                "message": ("No match for the residue name(s) given. "
                            f"Candidate ligand residues in this file: "
                            f"{', '.join(found)}" if found else
                            "No non-polymer residues found in this file."),
            })
        return jsonify({"sites": sites})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/receptors")
def api_receptors():
    return jsonify(get_all_receptors(RECEPTORS_DIR))


@app.route("/api/receptors/<name>/results")
def api_receptor_results(name: str):
    return jsonify(get_receptor_results(_receptor_dir(name)))


# ══════════════════════════════════════════════════════════════════════════════
# Docking
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/dock")
def dock_page():
    receptors = [r for r in get_all_receptors(RECEPTORS_DIR)
                 if r.get("n_ready", 0) > 0]
    ligands   = [l for l in get_library_ligands(LIBRARY_DIR)
                 if l.get("status") == "ready"]
    return render_template("dock.html",
                           receptors=receptors,
                           ligands=ligands,
                           jobs=get_job_queue(JOBS_DIR))


@app.route("/dock/submit", methods=["POST"])
def dock_submit():
    data          = request.get_json() or {}
    receptor_name = data.get("receptor")
    if not receptor_name:
        return jsonify({"error": "receptor required"}), 400
    rec_dir = _receptor_dir(receptor_name)
    if not rec_dir.exists():
        return jsonify({"error": f"receptor not found: {receptor_name}"}), 404
    try:
        job_id = submit_dock_job(
            JOBS_DIR,
            receptor_dir=str(rec_dir),
            library_dir=str(LIBRARY_DIR),
            options={
                "use_conformers": data.get("use_conformers", True),
                "force":          data.get("force", False),
                "nrun":           int(data.get("nrun", 20)),
                "max_failures":   int(data.get("max_failures", 5)),
                "max_atoms":      data.get("max_atoms"),
                "max_rot":        data.get("max_rot"),
                "compounds":      data.get("compounds"),
            }
        )
        return jsonify({"job_id": job_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ══════════════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/results/<receptor_name>")
def results_page(receptor_name: str):
    rec_dir = _receptor_dir(receptor_name)
    results = get_receptor_results(rec_dir)
    return render_template("results.html",
                           receptor_name=receptor_name,
                           results=results)


@app.route("/results/<receptor_name>/download/csv")
def download_csv(receptor_name: str):
    csv_path = _receptor_dir(receptor_name) / "results" / "results.csv"
    if not csv_path.exists():
        return "No results found", 404
    return send_file(csv_path, as_attachment=True,
                     download_name=f"{receptor_name}_results.csv")


@app.route("/results/<receptor_name>/download/complexes")
def download_complexes(receptor_name: str):
    complex_dir = _receptor_dir(receptor_name) / "results" / "complexes"
    if not complex_dir.exists():
        return "No complexes found", 404
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    shutil.make_archive(str(tmp_path.with_suffix("")), "zip", complex_dir)
    return send_file(tmp_path.with_suffix(".zip"), as_attachment=True,
                     download_name=f"{receptor_name}_complexes.zip")


@app.route("/results/<receptor_name>/complex/<ligand_name>")
def download_complex(receptor_name: str, ligand_name: str):
    safe = re.sub(r'[^a-zA-Z0-9_-]', '', ligand_name)
    path = (_receptor_dir(receptor_name) / "results" / "complexes"
            / f"complex_{safe}.pdb")
    if not path.exists():
        return "Complex not found", 404
    return send_file(path, as_attachment=True,
                     download_name=f"complex_{safe}.pdb")


# ══════════════════════════════════════════════════════════════════════════════
# SocketIO
# ══════════════════════════════════════════════════════════════════════════════

@sio.on("connect")
def on_connect():
    emit("status", {"connected": True})


@sio.on("request_status")
def on_request_status():
    emit("status_update", {
        "library": get_library_summary(LIBRARY_DIR),
        "jobs":    get_job_queue(JOBS_DIR),
    })


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    host  = os.environ.get("HOST",  "0.0.0.0")
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "").lower() in ("1", "true")
    print(f"AutoDock Web UI starting on {host}:{port}")
    sio.run(app, host=host, port=port, debug=debug,
            allow_unsafe_werkzeug=True)
