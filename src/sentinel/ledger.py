"""Append-only audit ledger using SQLite."""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path


class AuditLedger:
    """Records every scoring decision. Append-only, immutable."""

    def __init__(self, db_path: str = "data/audit.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                trans_num TEXT,
                amount REAL,
                risk_probability REAL,
                decision TEXT,
                expected_loss_allow REAL,
                expected_loss_block REAL,
                expected_loss_review REAL,
                model_version TEXT,
                latency_ms REAL,
                degraded INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

    def log(self, record: dict):
        """Write one decision to the ledger."""
        self.conn.execute(
            """INSERT OR IGNORE INTO decisions 
               (decision_id, timestamp, trans_num, amount, risk_probability,
                decision, expected_loss_allow, expected_loss_block,
                expected_loss_review, model_version, latency_ms, degraded)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record["decision_id"],
                datetime.now(timezone.utc).isoformat(),
                record.get("trans_num", ""),
                record["amount_inr"],
                record["risk_probability"],
                record["decision"],
                record["expected_loss_if_allowed_inr"],
                record["expected_loss_if_blocked_inr"],
                record["expected_loss_if_reviewed_inr"],
                record["model_version"],
                record["latency_ms"],
                1 if record.get("degraded", False) else 0,
            ),
        )
        self.conn.commit()

    def count(self) -> int:
        """Total decisions recorded."""
        row = self.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()
        return row[0]

    def get(self, decision_id: str) -> dict | None:
        """Retrieve a single decision by ID."""
        row = self.conn.execute(
            "SELECT * FROM decisions WHERE decision_id = ?", (decision_id,)
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in self.conn.execute("SELECT * FROM decisions LIMIT 0").description]
        return dict(zip(cols, row))