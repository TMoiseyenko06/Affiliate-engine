"""SQLite persistence layer for the Pinterest affiliate pipeline.

Default backend is SQLite (via ``DATABASE_URL=sqlite:///path.db``). A Postgres
DSN can be supplied instead; when psycopg is available the same schema/queries
run against Postgres. Everything else in the codebase talks to the ``Database``
class here rather than issuing SQL directly, so the storage engine is swappable.

Tables
------
posts          : one row per post attempt that reached the poster (or a skip)
performance    : analytics samples pulled periodically from Pinterest
products       : catalogue of candidate products and when each was last used
verifier_log   : every verifier verdict (pass/fail + reasons)
alerts         : dead-man's-switch and skip notifications
api_usage      : per-day per-provider call counters for budget caps
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS posts (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp          TEXT NOT NULL,
        content_type       TEXT NOT NULL,
        product_id_or_topic TEXT,
        board_id           TEXT,
        pin_id             TEXT,
        pin_url            TEXT,
        status             TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS performance (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id   INTEGER NOT NULL,
        saves     INTEGER,
        clicks    INTEGER,
        pulled_at TEXT NOT NULL,
        FOREIGN KEY (post_id) REFERENCES posts(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS products (
        product_id      TEXT PRIMARY KEY,
        title           TEXT,
        commission_rate REAL,
        last_used_at    TEXT,
        source          TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS verifier_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        post_attempt_id TEXT NOT NULL,
        pass            INTEGER NOT NULL,
        failures        TEXT,
        timestamp       TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS alerts (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        message   TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS api_usage (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        day      TEXT NOT NULL,
        provider TEXT NOT NULL,
        calls    INTEGER NOT NULL DEFAULT 0,
        UNIQUE(day, provider)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cycle_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        post_attempt_id TEXT NOT NULL,
        step            TEXT NOT NULL,
        status          TEXT NOT NULL,
        detail          TEXT,
        timestamp       TEXT NOT NULL
    )
    """,
]


def _sqlite_path_from_url(database_url: str) -> str:
    """Extract a filesystem path from a ``sqlite:///`` URL."""
    prefix = "sqlite:///"
    if database_url.startswith(prefix):
        return database_url[len(prefix):]
    if database_url.startswith("sqlite://"):
        return database_url[len("sqlite://"):]
    # Bare path fallback.
    return database_url


class Database:
    """Thin, thread-safe wrapper over a SQLite connection.

    A single connection is guarded by a lock; the pipeline is not
    high-concurrency (a handful of cron cycles a day), so this keeps the code
    simple while remaining safe if the analytics pull runs alongside a cycle.
    """

    def __init__(self, database_url: Optional[str] = None):
        from config import CONFIG

        self.database_url = database_url or CONFIG.database_url
        if not self.database_url.startswith("sqlite"):
            raise NotImplementedError(
                "Only sqlite is implemented in this build. Set DATABASE_URL to a "
                "sqlite:/// URL, or extend Database for Postgres."
            )
        self.path = _sqlite_path_from_url(self.database_url)
        self._lock = threading.RLock()
        # Allow the connection to be used from the analytics thread too.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.init_schema()

    # -- lifecycle ---------------------------------------------------------
    def init_schema(self) -> None:
        with self._lock:
            for stmt in SCHEMA_STATEMENTS:
                self._conn.execute(stmt)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _cursor(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            finally:
                cur.close()

    # -- posts -------------------------------------------------------------
    def create_post(
        self,
        content_type: str,
        product_id_or_topic: str,
        board_id: Optional[str],
        status: str,
        pin_id: Optional[str] = None,
        pin_url: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> int:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO posts
                    (timestamp, content_type, product_id_or_topic, board_id,
                     pin_id, pin_url, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _iso(timestamp or utcnow()),
                    content_type,
                    product_id_or_topic,
                    board_id,
                    pin_id,
                    pin_url,
                    status,
                ),
            )
            return int(cur.lastrowid)

    def update_post(
        self,
        post_id: int,
        *,
        status: Optional[str] = None,
        pin_id: Optional[str] = None,
        pin_url: Optional[str] = None,
    ) -> None:
        fields: List[str] = []
        values: List[Any] = []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if pin_id is not None:
            fields.append("pin_id = ?")
            values.append(pin_id)
        if pin_url is not None:
            fields.append("pin_url = ?")
            values.append(pin_url)
        if not fields:
            return
        values.append(post_id)
        with self._cursor() as cur:
            cur.execute(f"UPDATE posts SET {', '.join(fields)} WHERE id = ?", values)

    def recent_posts(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM posts ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def recent_affiliate_ratio(self, lookback_posts: int) -> float:
        """Fraction of the last ``lookback_posts`` *successful* posts that were
        affiliate. Returns 0.0 when there is no history yet."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT content_type FROM posts
                WHERE status = 'posted'
                ORDER BY id DESC LIMIT ?
                """,
                (lookback_posts,),
            )
            rows = cur.fetchall()
        if not rows:
            return 0.0
        affiliate = sum(1 for r in rows if r["content_type"] == "affiliate")
        return affiliate / len(rows)

    def last_successful_post_time(self) -> Optional[datetime]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT timestamp FROM posts WHERE status = 'posted' ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
        if not row:
            return None
        return datetime.fromisoformat(row["timestamp"])

    def topic_used_within(self, topic: str, days: int) -> bool:
        cutoff = _iso(utcnow() - timedelta(days=days))
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM posts
                WHERE product_id_or_topic = ?
                  AND status = 'posted'
                  AND timestamp >= ?
                LIMIT 1
                """,
                (topic, cutoff),
            )
            return cur.fetchone() is not None

    # -- products ----------------------------------------------------------
    def upsert_product(
        self,
        product_id: str,
        title: Optional[str] = None,
        commission_rate: Optional[float] = None,
        source: Optional[str] = None,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO products (product_id, title, commission_rate, source)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(product_id) DO UPDATE SET
                    title = COALESCE(excluded.title, products.title),
                    commission_rate = COALESCE(excluded.commission_rate, products.commission_rate),
                    source = COALESCE(excluded.source, products.source)
                """,
                (product_id, title, commission_rate, source),
            )

    def mark_product_used(self, product_id: str, when: Optional[datetime] = None) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE products SET last_used_at = ? WHERE product_id = ?",
                (_iso(when or utcnow()), product_id),
            )

    def product_used_within(self, product_id: str, days: int) -> bool:
        cutoff = utcnow() - timedelta(days=days)
        with self._cursor() as cur:
            cur.execute(
                "SELECT last_used_at FROM products WHERE product_id = ?",
                (product_id,),
            )
            row = cur.fetchone()
        if not row or not row["last_used_at"]:
            return False
        return datetime.fromisoformat(row["last_used_at"]) >= cutoff

    def products_used_within(self, days: int) -> List[str]:
        """Return product_ids used within the lookback window (for scout filtering)."""
        cutoff = _iso(utcnow() - timedelta(days=days))
        with self._cursor() as cur:
            cur.execute(
                "SELECT product_id FROM products WHERE last_used_at IS NOT NULL AND last_used_at >= ?",
                (cutoff,),
            )
            return [r["product_id"] for r in cur.fetchall()]

    # -- verifier log ------------------------------------------------------
    def log_verifier(self, post_attempt_id: str, passed: bool, failures: Iterable[str]) -> int:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO verifier_log (post_attempt_id, pass, failures, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (post_attempt_id, 1 if passed else 0, json.dumps(list(failures)), _iso(utcnow())),
            )
            return int(cur.lastrowid)

    # -- cycle step logging ------------------------------------------------
    def log_step(self, post_attempt_id: str, step: str, status: str, detail: Any = None) -> None:
        if detail is not None and not isinstance(detail, str):
            try:
                detail = json.dumps(detail, default=str)
            except (TypeError, ValueError):
                detail = str(detail)
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO cycle_log (post_attempt_id, step, status, detail, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (post_attempt_id, step, status, detail, _iso(utcnow())),
            )

    # -- performance -------------------------------------------------------
    def record_performance(self, post_id: int, saves: int, clicks: int) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO performance (post_id, saves, clicks, pulled_at)
                VALUES (?, ?, ?, ?)
                """,
                (post_id, saves, clicks, _iso(utcnow())),
            )

    def posts_with_pins(self) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM posts WHERE pin_id IS NOT NULL")
            return [dict(r) for r in cur.fetchall()]

    # -- alerts ------------------------------------------------------------
    def add_alert(self, message: str) -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO alerts (timestamp, message) VALUES (?, ?)",
                (_iso(utcnow()), message),
            )
            return int(cur.lastrowid)

    def recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    # -- api usage / budget ------------------------------------------------
    def increment_api_usage(self, provider: str, when: Optional[datetime] = None) -> int:
        day = (when or utcnow()).date().isoformat()
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO api_usage (day, provider, calls) VALUES (?, ?, 1)
                ON CONFLICT(day, provider) DO UPDATE SET calls = calls + 1
                """,
                (day, provider),
            )
            cur.execute(
                "SELECT calls FROM api_usage WHERE day = ? AND provider = ?",
                (day, provider),
            )
            row = cur.fetchone()
            return int(row["calls"]) if row else 0

    def api_usage_today(self, provider: str, when: Optional[datetime] = None) -> int:
        day = (when or utcnow()).date().isoformat()
        with self._cursor() as cur:
            cur.execute(
                "SELECT calls FROM api_usage WHERE day = ? AND provider = ?",
                (day, provider),
            )
            row = cur.fetchone()
            return int(row["calls"]) if row else 0


_DEFAULT_DB: Optional[Database] = None


def get_db(database_url: Optional[str] = None) -> Database:
    """Return a process-wide default Database instance (lazy singleton)."""
    global _DEFAULT_DB
    if _DEFAULT_DB is None or database_url is not None:
        _DEFAULT_DB = Database(database_url)
    return _DEFAULT_DB
