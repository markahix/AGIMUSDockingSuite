"""
job_writer.py — write job request files to jobs/pending/.
All paths are absolute host paths. Containers mirror them exactly.
"""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path

_SAFE_PATH = re.compile(r'^[a-zA-Z0-9/_.\-]+$')


def _safe_path(value: str, label: str) -> str:
    p = str(value)
    if not p.startswith("/"):
        raise ValueError(f"{label} must be absolute: {p!r}")
    if not _SAFE_PATH.match(p):
        raise ValueError(f"{label} contains unsafe characters: {p!r}")
    return p


def _write(jobs_dir: Path, job: dict) -> str:
    pending = jobs_dir / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    (pending / f"{job['job_id']}.json").write_text(json.dumps(job, indent=2))
    return job["job_id"]


def submit_prep_ligands_job(jobs_dir: Path, library_dir: str,
                             ligand_dir: str, options: dict = None) -> str:
    opts = options or {}
    job_id = f"prep_lig_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    job = {
        "job_id":      job_id,
        "type":        "prep_ligands",
        "library_dir": _safe_path(library_dir, "library_dir"),
        "ligand_dir":  _safe_path(ligand_dir,  "ligand_dir"),
        "options": {
            "refresh_all": bool(opts.get("refresh_all", False)),
            "conformers":  (int(opts["conformers"])
                            if opts.get("conformers") is not None else None),
        },
        "submitted_at": datetime.now().isoformat(),
        "submitted_by": "webui",
    }
    return _write(jobs_dir, job)


def submit_prep_receptor_job(jobs_dir: Path, pdb_path: str,
                              receptors_dir: str, library_dir: str,
                              center_x: float, center_y: float, center_z: float,
                              padding: float = 10.0,
                              rmsd_threshold: float = 0.5,
                              size_x: float = None,
                              size_y: float = None,
                              size_z: float = None,
                              remove_resnames: list = None,
                              keep_water: bool = False,
                              box_mode: str = None,
                              apron_reach: float = 8.0,
                              metadata: dict = None) -> str:
    """
    metadata dict: name, variant, gene_id, gene_db, notes
    Receptor directory name is derived from the PDB filename (already sanitized).
    """
    job_id = f"prep_rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    job = {
        "job_id":        job_id,
        "type":          "prep_receptor",
        "pdb_path":      _safe_path(pdb_path,      "pdb_path"),
        "receptors_dir": _safe_path(receptors_dir, "receptors_dir"),
        "library_dir":   _safe_path(library_dir,   "library_dir"),
        "options": {
            "center_x":        float(center_x),
            "center_y":        float(center_y),
            "center_z":        float(center_z),
            "size_x":          float(size_x) if size_x is not None else None,
            "size_y":          float(size_y) if size_y is not None else None,
            "size_z":          float(size_z) if size_z is not None else None,
            "padding":         float(padding),
            "rmsd_threshold":  float(rmsd_threshold),
            "remove_resnames": remove_resnames,
            "keep_water":      bool(keep_water),
            "box_mode":        box_mode,
            "apron_reach":     float(apron_reach),
        },
        "metadata": metadata or {},
        "submitted_at": datetime.now().isoformat(),
        "submitted_by": "webui",
    }
    return _write(jobs_dir, job)


def submit_dock_job(jobs_dir: Path, receptor_dir: str, library_dir: str,
                    options: dict = None) -> str:
    opts = options or {}
    job_id = f"dock_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # Validate compounds list
    compounds = None
    if opts.get("compounds"):
        compounds = [c for c in opts["compounds"]
                     if re.match(r'^[a-zA-Z0-9_\-]+$', str(c))]

    job = {
        "job_id":       job_id,
        "type":         "dock",
        "receptor_dir": _safe_path(receptor_dir, "receptor_dir"),
        "library_dir":  _safe_path(library_dir,  "library_dir"),
        "options": {
            "use_conformers": bool(opts.get("use_conformers", True)),
            "force":          bool(opts.get("force", False)),
            "nrun":           int(opts.get("nrun", 20)),
            "max_failures":   int(opts.get("max_failures", 5)),
            "max_atoms":      (int(opts["max_atoms"]) if opts.get("max_atoms") else None),
            "max_rot":        (int(opts["max_rot"])   if opts.get("max_rot")   else None),
            "compounds":      compounds,
        },
        "submitted_at": datetime.now().isoformat(),
        "submitted_by": "webui",
    }
    return _write(jobs_dir, job)
