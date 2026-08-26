"""Velocity store — tracks per-card transaction history for real-time features."""

import sqlite3
import time
from pathlib import Path


class VelocityStore:
    """SQLite-backed velocity counters per card."""

    def __init__(self, db_path: str = "data/velocity.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_table()
        self._available = True

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS card_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                card_id TEXT NOT NULL,
                unix_time INTEGER NOT NULL,
                amount REAL NOT NULL,
                merchant TEXT NOT NULL
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_card_time ON card_transactions(card_id, unix_time)"
        )
        self.conn.commit()

    @property
    def is_available(self) -> bool:
        return self._available

    def get_history(self, card_id: str, current_time: int) -> dict | None:
        """Get velocity features for a card at a point in time."""
        try:
            # Transactions in last 1 hour
            one_hour_ago = current_time - 3600
            twenty_four_hours_ago = current_time - 86400

            rows = self.conn.execute(
                "SELECT unix_time, amount, merchant FROM card_transactions WHERE card_id = ? AND unix_time > ?",
                (str(card_id), twenty_four_hours_ago),
            ).fetchall()

            if not rows:
                return None

            count_1h = sum(1 for r in rows if r[0] > one_hour_ago)
            count_24h = len(rows)
            sum_24h = sum(r[1] for r in rows)
            amounts = [r[1] for r in rows]
            merchants_24h = len(set(r[2] for r in rows))

            import numpy as np
            return {
                "txn_count_1h": count_1h,
                "txn_count_24h": count_24h,
                "txn_sum_24h": sum_24h,
                "median_amt": float(np.median(amounts)),
                "distinct_merchants_24h": merchants_24h,
            }

        except Exception:
            self._available = False
            return None

    def record(self, card_id: str, unix_time: int, amount: float, merchant: str):
        """Record a transaction for future velocity lookups."""
        try:
            self.conn.execute(
                "INSERT INTO card_transactions (card_id, unix_time, amount, merchant) VALUES (?, ?, ?, ?)",
                (str(card_id), unix_time, amount, merchant),
            )
            self.conn.commit()
        except Exception:
            self._available = False

    def cleanup(self, older_than_seconds: int = 2592000):
        """Remove entries older than 30 days."""
        cutoff = int(time.time()) - older_than_seconds
        self.conn.execute("DELETE FROM card_transactions WHERE unix_time < ?", (cutoff,))
        self.conn.commit()


class DegradedVelocityStore:
    """Fallback when real store is unavailable. Returns None for all lookups."""

    @property
    def is_available(self) -> bool:
        return False

    def get_history(self, card_id: str, current_time: int) -> dict | None:
        return None

    def record(self, card_id: str, unix_time: int, amount: float, merchant: str):
        pass