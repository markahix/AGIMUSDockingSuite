"""
library_state.py
----------------
SQLite-backed state management for the ligand library.
Uses WAL mode for crash recovery and concurrent read safety.

Status values:
  pending    — PDB present, not yet processed
  processing — currently being processed (lock held)
  ready      — fully processed, available for docking
  failed     — processing failed, error_msg set

All writes use transactions. Processing lock is claimed by writing
status='processing' inside a transaction — only one writer wins.
Stale 'processing' records (older than STALE_MINUTES) are reset to
'pending' on startup.
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

STALE_MINUTES = 30  # processing records older than this are assumed crashed

SCHEMA = """
CREATE TABLE IF NOT EXISTS ligands (
    name                TEXT PRIMARY KEY,
    pdb_hash            TEXT,
    status              TEXT NOT NULL DEFAULT 'pending',
    pdbqt_path          TEXT,
    n_conformers        INTEGER DEFAULT 0,
    conf_dir            TEXT,
    use_flex_fallback   INTEGER DEFAULT 0,
    n_heavy_atoms       INTEGER DEFAULT 0,
    n_hydrogens         INTEGER DEFAULT 0,
    n_rot_bonds         INTEGER DEFAULT 0,
    formula             TEXT,
    smiles              TEXT,
    smiles_source       TEXT,
    bbox_json           TEXT,
    elements_json       TEXT,
    error_msg           TEXT,
    last_updated        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_status ON ligands(status);
CREATE INDEX IF NOT EXISTS idx_last_updated ON ligands(last_updated);
"""


@contextmanager
def _conn(db_path: Path):
    """Open a WAL-mode SQLite connection, ensure schema, yield connection."""
    con = sqlite3.connect(str(db_path), timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    try:
        yield con
    finally:
        con.close()


class LibraryState:
    """
    Interface to the library SQLite state database.
    One instance per process; db_path is typically library/library_state.db.
    """

    def __init__(self, library_dir: Path):
        self.library_dir = Path(library_dir)
        self.db_path     = self.library_dir / "library_state.db"
        self.library_dir.mkdir(parents=True, exist_ok=True)
        # Initialize schema
        with _conn(self.db_path):
            pass
        self._reset_stale()

    def _now(self) -> str:
        return datetime.utcnow().isoformat()

    def _reset_stale(self):
        """Reset any processing records that are older than STALE_MINUTES."""
        cutoff = (datetime.utcnow() - timedelta(minutes=STALE_MINUTES)).isoformat()
        with _conn(self.db_path) as con:
            cur = con.execute(
                "UPDATE ligands SET status='pending', error_msg='reset after stale lock' "
                "WHERE status='processing' AND last_updated < ?",
                (cutoff,)
            )
            con.commit()
            if cur.rowcount:
                print(f"[state] Reset {cur.rowcount} stale processing record(s)")

    def register(self, name: str, pdb_hash: str) -> bool:
        """
        Register a ligand as pending if not already known.
        If known with same hash, does nothing and returns False.
        If known with different hash, resets to pending for reprocessing.
        Returns True if registration created/updated a record.
        """
        with _conn(self.db_path) as con:
            row = con.execute(
                "SELECT name, pdb_hash, status FROM ligands WHERE name=?", (name,)
            ).fetchone()

            if row is None:
                con.execute(
                    "INSERT INTO ligands (name, pdb_hash, status, last_updated) "
                    "VALUES (?, ?, 'pending', ?)",
                    (name, pdb_hash, self._now())
                )
                con.commit()
                return True

            if row["pdb_hash"] != pdb_hash:
                # File changed — reset for reprocessing
                con.execute(
                    "UPDATE ligands SET pdb_hash=?, status='pending', "
                    "error_msg=NULL, last_updated=? WHERE name=?",
                    (pdb_hash, self._now(), name)
                )
                con.commit()
                return True

            return False  # already known, same hash

    def claim(self, name: str) -> bool:
        """
        Atomically claim a pending ligand for processing.
        Returns True if claim succeeded, False if already claimed/processed.
        """
        with _conn(self.db_path) as con:
            cur = con.execute(
                "UPDATE ligands SET status='processing', last_updated=? "
                "WHERE name=? AND status='pending'",
                (self._now(), name)
            )
            con.commit()
            return cur.rowcount == 1

    def mark_ready(self, name: str, data: dict):
        """Mark a ligand as ready with full metadata."""
        with _conn(self.db_path) as con:
            con.execute("""
                UPDATE ligands SET
                    status            = 'ready',
                    pdbqt_path        = :pdbqt_path,
                    n_conformers      = :n_conformers,
                    conf_dir          = :conf_dir,
                    use_flex_fallback = :use_flex_fallback,
                    n_heavy_atoms     = :n_heavy_atoms,
                    n_hydrogens       = :n_hydrogens,
                    n_rot_bonds       = :n_rot_bonds,
                    formula           = :formula,
                    smiles            = :smiles,
                    smiles_source     = :smiles_source,
                    bbox_json         = :bbox_json,
                    elements_json     = :elements_json,
                    error_msg         = NULL,
                    last_updated      = :last_updated
                WHERE name = :name
            """, {**data, "name": name, "last_updated": self._now()})
            con.commit()

    def mark_failed(self, name: str, error: str):
        """Mark a ligand as failed with an error message."""
        with _conn(self.db_path) as con:
            con.execute(
                "UPDATE ligands SET status='failed', error_msg=?, "
                "last_updated=? WHERE name=?",
                (error[:500], self._now(), name)
            )
            con.commit()

    def get(self, name: str) -> dict | None:
        """Return full record for a ligand, or None if not found."""
        with _conn(self.db_path) as con:
            row = con.execute(
                "SELECT * FROM ligands WHERE name=?", (name,)
            ).fetchone()
            return dict(row) if row else None

    def get_ready(self) -> list[dict]:
        """Return all ligands with status='ready'."""
        with _conn(self.db_path) as con:
            rows = con.execute(
                "SELECT * FROM ligands WHERE status='ready' ORDER BY name"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_pending(self) -> list[str]:
        """Return names of all pending ligands."""
        with _conn(self.db_path) as con:
            rows = con.execute(
                "SELECT name FROM ligands WHERE status='pending' ORDER BY name"
            ).fetchall()
            return [r["name"] for r in rows]

    def summary(self) -> dict:
        """Return count of ligands in each status."""
        with _conn(self.db_path) as con:
            rows = con.execute(
                "SELECT status, COUNT(*) as n FROM ligands GROUP BY status"
            ).fetchall()
            return {r["status"]: r["n"] for r in rows}

    def export_json(self) -> dict:
        """
        Export all ready ligands as a dict compatible with the legacy
        library.json format, for backward compatibility with dock_batch.py
        during transition.
        """
        out = {}
        for row in self.get_ready():
            bbox   = json.loads(row["bbox_json"])     if row["bbox_json"]     else {}
            elems  = json.loads(row["elements_json"]) if row["elements_json"] else []
            out[row["name"]] = {
                "name":             row["name"],
                "formula":          row["formula"],
                "smiles":           row["smiles"],
                "smiles_source":    row["smiles_source"],
                "pdbqt":            row["pdbqt_path"],
                "n_heavy_atoms":    row["n_heavy_atoms"],
                "n_hydrogens":      row["n_hydrogens"],
                "n_rot_bonds":      row["n_rot_bonds"],
                "n_conformers":     row["n_conformers"],
                "conf_dir":         row["conf_dir"],
                "use_flex_fallback": bool(row["use_flex_fallback"]),
                "bbox":             bbox,
                "elements":         elems,
                "status":           "ok",
            }
        return out

    def write_json(self, path: Path):
        """Write legacy library.json for backward compatibility."""
        path.write_text(json.dumps(self.export_json(), indent=2))
