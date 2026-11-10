"""
receptor_state.py
-----------------
SQLite-backed state management for a single receptor directory.
Each receptor gets its own receptor_state.db.

Tables:
  receptor_info  — key/value metadata for this receptor
  models         — one row per structural model
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

STALE_MINUTES = 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS receptor_info (
    key     TEXT PRIMARY KEY,
    value   TEXT
);

CREATE TABLE IF NOT EXISTS models (
    model_id            INTEGER PRIMARY KEY,
    model_dir           TEXT NOT NULL,
    source_file         TEXT,
    prep_status         TEXT NOT NULL DEFAULT 'pending',
    active_site_rmsd    REAL DEFAULT -1.0,
    included            INTEGER DEFAULT 1,
    exclusion_reason    TEXT,
    last_updated        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_model_status ON models(prep_status);
"""


@contextmanager
def _conn(db_path: Path):
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), timeout=30)
    con.row_factory = sqlite3.Row
    # Use DELETE journal mode — receptor prep is single-process sequential,
    # no concurrent readers needed, avoids WAL sidecar file permission issues
    con.execute("PRAGMA journal_mode=DELETE")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    try:
        yield con
    finally:
        con.close()


class ReceptorState:
    """State manager for a single receptor directory."""

    def __init__(self, receptor_dir: Path):
        self.receptor_dir = Path(receptor_dir)
        self.db_path      = self.receptor_dir / "receptor_state.db"
        self.receptor_dir.mkdir(parents=True, exist_ok=True)
        with _conn(self.db_path):
            pass
        self._reset_stale()

    def _now(self) -> str:
        return datetime.utcnow().isoformat()

    def _reset_stale(self):
        cutoff = (datetime.utcnow() - timedelta(minutes=STALE_MINUTES)).isoformat()
        with _conn(self.db_path) as con:
            cur = con.execute(
                "UPDATE models SET prep_status='pending' "
                "WHERE prep_status='processing' AND last_updated < ?",
                (cutoff,)
            )
            con.commit()
            if cur.rowcount:
                print(f"[receptor_state] Reset {cur.rowcount} stale model(s)")

    # ── Receptor info ──────────────────────────────────────────────────────────

    def set_info(self, **kwargs):
        with _conn(self.db_path) as con:
            for k, v in kwargs.items():
                con.execute(
                    "INSERT OR REPLACE INTO receptor_info(key,value) VALUES(?,?)",
                    (k, json.dumps(v) if not isinstance(v, str) else v)
                )
            con.commit()

    def get_info(self, key: str, default=None):
        with _conn(self.db_path) as con:
            row = con.execute(
                "SELECT value FROM receptor_info WHERE key=?", (key,)
            ).fetchone()
            if row is None:
                return default
            try:
                return json.loads(row["value"])
            except Exception:
                return row["value"]

    # ── Model management ───────────────────────────────────────────────────────

    def add_model(self, model_id: int, model_dir: str,
                  source_file: str = None) -> bool:
        """Register a new model. Returns True if inserted, False if exists."""
        with _conn(self.db_path) as con:
            existing = con.execute(
                "SELECT model_id FROM models WHERE model_id=?", (model_id,)
            ).fetchone()
            if existing:
                return False
            con.execute(
                "INSERT INTO models(model_id, model_dir, source_file, "
                "prep_status, last_updated) VALUES(?,?,?,'pending',?)",
                (model_id, model_dir, source_file, self._now())
            )
            con.commit()
            return True

    def claim_model(self, model_id: int) -> bool:
        with _conn(self.db_path) as con:
            cur = con.execute(
                "UPDATE models SET prep_status='processing', last_updated=? "
                "WHERE model_id=? AND prep_status='pending'",
                (self._now(), model_id)
            )
            con.commit()
            return cur.rowcount == 1

    def mark_model_ready(self, model_id: int, active_site_rmsd: float = -1.0):
        with _conn(self.db_path) as con:
            con.execute(
                "UPDATE models SET prep_status='ready', "
                "active_site_rmsd=?, last_updated=? WHERE model_id=?",
                (active_site_rmsd, self._now(), model_id)
            )
            con.commit()

    def mark_model_failed(self, model_id: int, reason: str):
        with _conn(self.db_path) as con:
            con.execute(
                "UPDATE models SET prep_status='failed', included=0, "
                "exclusion_reason=?, last_updated=? WHERE model_id=?",
                (reason[:500], self._now(), model_id)
            )
            con.commit()

    def exclude_model(self, model_id: int, reason: str):
        """Mark a model as excluded (e.g. active site RMSD too similar)."""
        with _conn(self.db_path) as con:
            con.execute(
                "UPDATE models SET included=0, exclusion_reason=?, "
                "last_updated=? WHERE model_id=?",
                (reason, self._now(), model_id)
            )
            con.commit()

    def get_ready_models(self) -> list[dict]:
        with _conn(self.db_path) as con:
            rows = con.execute(
                "SELECT * FROM models WHERE prep_status='ready' "
                "AND included=1 ORDER BY model_id"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_models(self) -> list[dict]:
        with _conn(self.db_path) as con:
            rows = con.execute(
                "SELECT * FROM models ORDER BY model_id"
            ).fetchall()
            return [dict(r) for r in rows]

    def summary(self) -> dict:
        with _conn(self.db_path) as con:
            rows = con.execute(
                "SELECT prep_status, included, COUNT(*) as n "
                "FROM models GROUP BY prep_status, included"
            ).fetchall()
            return [dict(r) for r in rows]
