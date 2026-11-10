"""
state_reader.py
---------------
Read-only access to state databases from all three containers.
The web UI never writes to library or receptor state — it only reads.
All writes go through job files (for docking) or file deposits
(for ligands/receptors).
"""

import json
import sqlite3
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# Library state
# ══════════════════════════════════════════════════════════════════════════════

def get_library_summary(library_dir: Path) -> dict:
    """Return counts by status from library_state.db."""
    db = library_dir / "library_state.db"
    if not db.exists():
        return {"ready": 0, "pending": 0, "processing": 0,
                "failed": 0, "total": 0, "available": False}
    try:
        con  = sqlite3.connect(str(db), timeout=5)
        con.execute("PRAGMA query_only=ON")
        rows = con.execute(
            "SELECT status, COUNT(*) as n FROM ligands GROUP BY status"
        ).fetchall()
        con.close()
        counts = {r[0]: r[1] for r in rows}
        counts["total"]     = sum(counts.values())
        counts["available"] = True
        return counts
    except Exception as e:
        return {"available": False, "total": 0, "error": str(e)}


def get_library_ligands(library_dir: Path) -> list[dict]:
    """Return all ligands with their full metadata."""
    db = library_dir / "library_state.db"
    if not db.exists():
        # Fall back to legacy library.json
        j = library_dir / "library.json"
        if j.exists():
            try:
                lib = json.loads(j.read_text())
                return list(lib.values())
            except Exception:
                pass
        return []
    try:
        con = sqlite3.connect(str(db), timeout=5)
        con.execute("PRAGMA query_only=ON")
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM ligands ORDER BY name"
        ).fetchall()
        con.close()
        out = []
        for row in rows:
            d = dict(row)
            d["bbox"]     = json.loads(d.get("bbox_json")     or "{}")
            d["elements"] = json.loads(d.get("elements_json") or "[]")
            out.append(d)
        return out
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Receptor state
# ══════════════════════════════════════════════════════════════════════════════

def get_all_receptors(receptors_dir: Path) -> list[dict]:
    """Return summary info for all receptor directories."""
    receptors = []
    if not receptors_dir.exists():
        return receptors

    for state_db in sorted(receptors_dir.glob("*/receptor_state.db")):
        receptor_dir  = state_db.parent
        receptor_name = receptor_dir.name
        if receptor_name in ("inbox", "processed", "failed"):
            continue

        info = {"name": receptor_name, "available": False,
                "n_models": 0, "n_ready": 0, "has_results": False}

        try:
            con = sqlite3.connect(str(state_db), timeout=5)
            con.execute("PRAGMA query_only=ON")
            con.row_factory = sqlite3.Row

            # Get receptor info
            rows = con.execute("SELECT key, value FROM receptor_info").fetchall()
            for row in rows:
                try:
                    info[row["key"]] = json.loads(row["value"])
                except Exception:
                    info[row["key"]] = row["value"]

            # Get model counts
            model_rows = con.execute(
                "SELECT prep_status, included, COUNT(*) as n "
                "FROM models GROUP BY prep_status, included"
            ).fetchall()
            total = ready = 0
            for mr in model_rows:
                total += mr["n"]
                if mr["prep_status"] == "ready" and mr["included"]:
                    ready += mr["n"]
            info["n_models"] = total
            info["n_ready"]  = ready
            info["available"] = True
            con.close()
        except Exception:
            pass

        # Check for results
        info["has_results"] = (receptor_dir / "results" / "results.csv").exists()
        receptors.append(info)

    return receptors


def get_receptor_models(receptor_dir: Path) -> list[dict]:
    """Return all models for a receptor."""
    db = receptor_dir / "receptor_state.db"
    if not db.exists():
        return []
    try:
        con = sqlite3.connect(str(db), timeout=5)
        con.execute("PRAGMA query_only=ON")
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM models ORDER BY model_id"
        ).fetchall()
        con.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_receptor_results(receptor_dir: Path) -> list[dict]:
    """Return results from results.csv for a receptor."""
    import csv
    csv_path = receptor_dir / "results" / "results.csv"
    if not csv_path.exists():
        return []
    try:
        with open(csv_path) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Job queue state
# ══════════════════════════════════════════════════════════════════════════════

def get_job_queue(jobs_dir: Path) -> dict:
    """Return counts and recent jobs from each queue directory."""
    out = {}
    for state in ("pending", "running", "completed", "failed"):
        d     = jobs_dir / state
        files = sorted(d.glob("*.json")) if d.exists() else []
        jobs  = []
        for f in files[-10:]:  # show last 10
            try:
                jobs.append(json.loads(f.read_text()))
            except Exception:
                pass
        out[state] = {"count": len(files), "recent": jobs}
    return out


def _job_item_count(job: dict) -> int:
    """
    Best-effort count of work units in a job, for scaling runtime estimates.

    Input:  job - a job dict
    Output: number of ligands named (dock with an explicit compound list),
            else 0 (unknown — e.g. a whole-library dock or a prep job)
    """
    opts = job.get("options", {}) or {}
    c = opts.get("compounds")
    return len(c) if isinstance(c, list) else 0


def get_queue_view(jobs_dir: Path, recent: int = 20) -> dict:
    """
    Flat, display-ready view of the job queue for the Queue page.

    Input:  jobs_dir - the data/jobs directory
            recent   - how many finished jobs of each kind to keep
    Output: {"rows": [...], "counts": {state: n}} where each row has job_id,
            type, status, submitted_at, started_at, completed_at, host,
            elapsed_s (running or finished), expected_s (pending/running — a
            rough median of completed jobs of the same type)
    Behaviour: running and pending jobs are shown in queue order; finished jobs
            are capped to the most recent `recent` each, newest first. The
            runtime estimate is historical and approximate — AutoDock timing
            varies widely with pocket, library and GPU contention.
    """
    from datetime import datetime
    import statistics

    def _p(ts):
        try:
            return datetime.fromisoformat(ts) if ts else None
        except (ValueError, TypeError):
            return None

    buckets = {s: [] for s in ("running", "pending", "completed", "failed")}
    durations = {}
    for state in buckets:
        d = jobs_dir / state
        for f in (sorted(d.glob("*.json")) if d.exists() else []):
            try:
                j = json.loads(f.read_text())
            except Exception:
                continue
            buckets[state].append(j)
            if state == "completed":
                st = _p(j.get("started_at")) or _p(j.get("submitted_at"))
                ct = _p(j.get("completed_at"))
                if st and ct and ct >= st:
                    durations.setdefault(j.get("type"), []).append(
                        (ct - st).total_seconds())

    est = {t: statistics.median(v) for t, v in durations.items() if v}
    now = datetime.now()

    def _row(j, state):
        st, ct = _p(j.get("started_at")), _p(j.get("completed_at"))
        elapsed = None
        if state == "running" and st:
            elapsed = (now - st).total_seconds()
        elif state in ("completed", "failed") and st and ct and ct >= st:
            elapsed = (ct - st).total_seconds()
        expected = est.get(j.get("type")) if state in ("pending", "running") else None
        return {
            "job_id":       j.get("job_id", ""),
            "type":         j.get("type", "?"),
            "status":       state,
            "submitted_at": j.get("submitted_at"),
            "started_at":   j.get("started_at"),
            "completed_at": j.get("completed_at"),
            "host":         j.get("host"),
            "elapsed_s":    round(elapsed) if elapsed is not None else None,
            "expected_s":   round(expected) if expected is not None else None,
        }

    rows = []
    for j in sorted(buckets["running"], key=lambda x: x.get("started_at") or ""):
        rows.append(_row(j, "running"))
    for j in sorted(buckets["pending"], key=lambda x: x.get("submitted_at") or ""):
        rows.append(_row(j, "pending"))
    for j in sorted(buckets["failed"],
                    key=lambda x: x.get("completed_at") or x.get("submitted_at") or "",
                    reverse=True)[:recent]:
        rows.append(_row(j, "failed"))
    for j in sorted(buckets["completed"],
                    key=lambda x: x.get("completed_at") or x.get("submitted_at") or "",
                    reverse=True)[:recent]:
        rows.append(_row(j, "completed"))

    return {"rows": rows, "counts": {s: len(buckets[s]) for s in buckets}}


def get_inbox_contents(receptors_dir: Path) -> list[str]:
    """Return PDB filenames currently in the receptor inbox."""
    inbox = receptors_dir / "inbox"
    if not inbox.exists():
        return []
    return sorted(f.name for f in inbox.glob("*.pdb"))
