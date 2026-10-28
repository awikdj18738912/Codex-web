"""Crash-safe, resumable SQLite ledger for locally generated examples."""
from __future__ import annotations

import json
import sqlite3
import threading
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def normalize_source(value: str) -> str:
    """Normalize spacing and Unicode while retaining punctuation distinctions."""
    return " ".join(unicodedata.normalize("NFKC", value).split())


class Ledger:
    def __init__(self, path, quotas, config_hash):
        self.path = str(path)
        self.quotas = {str(k): int(v) for k, v in quotas.items()}
        if any(v < 0 for v in self.quotas.values()):
            raise ValueError("quotas must be non-negative")
        self.config_hash = str(config_hash)
        self._lock = threading.RLock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS samples (
                    id TEXT PRIMARY KEY, category TEXT NOT NULL, group_id TEXT,
                    source TEXT NOT NULL, source_norm TEXT NOT NULL, target TEXT NOT NULL,
                    payload_json TEXT NOT NULL, status TEXT NOT NULL, reason TEXT,
                    source_fingerprint TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS samples_source_norm ON samples(source_norm);
                CREATE INDEX IF NOT EXISTS samples_category_status ON samples(category,status);
                CREATE TABLE IF NOT EXISTS counters (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS requests (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, tag TEXT NOT NULL,
                    request_json TEXT NOT NULL, response_json TEXT, error TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_pending (
                    id TEXT PRIMARY KEY, category TEXT NOT NULL,
                    payload_json TEXT NOT NULL, updated_at TEXT NOT NULL
                );
            """)
            expected = {"config_hash": self.config_hash,
                        "quotas": json.dumps(self.quotas, sort_keys=True, separators=(",", ":"))}
            current = dict(db.execute("SELECT key,value FROM metadata"))
            if current and any(current.get(k) != v for k, v in expected.items()):
                raise ValueError("ledger_config_mismatch")
            db.executemany("INSERT OR IGNORE INTO metadata(key,value) VALUES(?,?)", expected.items())
            db.execute("INSERT OR IGNORE INTO counters(key,value) VALUES('sequence',0)")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA busy_timeout=60000")
            yield db
        finally:
            db.close()

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    def record_candidate(self, id, category, group_id, source, target, payload,
                         verdict, rejection_reason, source_fingerprint=None) -> str:
        source_norm = normalize_source(source)
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        now = self._now()
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT status FROM samples WHERE id=?", (str(id),)).fetchone()
            if old:
                db.execute("DELETE FROM audit_pending WHERE id=?", (str(id),))
                db.commit()
                return old["status"]
            if verdict != "passed":
                status, reason = "rejected", rejection_reason or str(verdict)
            else:
                peers = db.execute("SELECT id,target,status FROM samples WHERE source_norm=? AND status IN ('accepted','conflict')", (source_norm,)).fetchall()
                if peers and any(p["target"] != target for p in peers):
                    # A conflicting target invalidates every accepted example for this source.
                    db.execute("UPDATE samples SET status='conflict',reason='source_target_conflict',updated_at=? WHERE source_norm=? AND status='accepted'", (now, source_norm))
                    status, reason = "conflict", "source_target_conflict"
                elif peers:
                    status, reason = "duplicate", "duplicate_source"
                else:
                    used = db.execute("SELECT COUNT(*) FROM samples WHERE category=? AND status='accepted'", (category,)).fetchone()[0]
                    if category not in self.quotas or used >= self.quotas[category]:
                        status, reason = "quota_full", "category_quota_full"
                    else:
                        status, reason = "accepted", None
            db.execute("""INSERT INTO samples(id,category,group_id,source,source_norm,target,payload_json,status,reason,source_fingerprint,created_at,updated_at)
                          VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                       (str(id), str(category), group_id, source, source_norm, target, payload_json,
                        status, reason, source_fingerprint, now, now))
            db.execute("DELETE FROM audit_pending WHERE id=?", (str(id),))
            db.commit()
            return status

    def defer_audit(self, id, payload):
        """Persist an unaudited candidate so a transient audit failure is resumable."""
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        category = str(payload.get("category", "")) if isinstance(payload, dict) else ""
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM samples WHERE id=?", (str(id),)).fetchone():
                db.execute("DELETE FROM audit_pending WHERE id=?", (str(id),))
            else:
                db.execute("""INSERT INTO audit_pending(id,category,payload_json,updated_at)
                              VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                              category=excluded.category,payload_json=excluded.payload_json,
                              updated_at=excluded.updated_at""",
                           (str(id), category, payload_json, self._now()))
            db.commit()

    def iter_pending_audits(self, category=None):
        with self._connect() as db:
            if category is None:
                cursor = db.execute("SELECT id,payload_json FROM audit_pending ORDER BY rowid")
            else:
                cursor = db.execute("SELECT id,payload_json FROM audit_pending WHERE category=? ORDER BY rowid", (category,))
            try:
                while True:
                    rows = cursor.fetchmany(256)
                    if not rows:
                        break
                    for row in rows:
                        payload = json.loads(row["payload_json"])
                        # Preserve the candidate id if an incomplete payload omitted it.
                        payload.setdefault("id", row["id"])
                        yield payload
            finally:
                cursor.close()

    def stats(self):
        with self._connect() as db:
            rows = db.execute("SELECT category,status,COUNT(*) n FROM samples GROUP BY category,status").fetchall()
            pending_rows = db.execute("SELECT category,COUNT(*) n FROM audit_pending GROUP BY category").fetchall()
        result = {c: {"quota": q, "accepted": 0, "rejected": 0, "duplicate": 0,
                      "conflict": 0, "quota_full": 0, "pending_audit": 0, "total": 0} for c, q in self.quotas.items()}
        for row in rows:
            bucket = result.setdefault(row["category"], {"quota": 0, "accepted": 0, "rejected": 0, "duplicate": 0, "conflict": 0, "quota_full": 0, "pending_audit": 0, "total": 0})
            bucket[row["status"]] = row["n"]
        for row in pending_rows:
            bucket = result.setdefault(row["category"], {"quota": 0, "accepted": 0, "rejected": 0, "duplicate": 0, "conflict": 0, "quota_full": 0, "pending_audit": 0, "total": 0})
            bucket["pending_audit"] = row["n"]
        for bucket in result.values():
            bucket["total"] = sum(bucket[k] for k in ("accepted", "rejected", "duplicate", "conflict", "quota_full"))
        result["total"] = {k: sum(v[k] for c, v in result.items() if c != "total")
                            for k in ("quota", "accepted", "rejected", "duplicate", "conflict", "quota_full", "pending_audit", "total")}
        return result

    def iter_accepted(self, category=None):
        with self._connect() as db:
            if category is None:
                cursor = db.execute("SELECT * FROM samples WHERE status='accepted' ORDER BY rowid")
            else:
                cursor = db.execute("SELECT * FROM samples WHERE status='accepted' AND category=? ORDER BY rowid", (category,))
            try:
                while True:
                    rows = cursor.fetchmany(256)
                    if not rows:
                        break
                    for row in rows:
                        item = dict(row)
                        item["payload"] = json.loads(item.pop("payload_json"))
                        yield item
            finally:
                cursor.close()

    def next_sequence(self):
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE counters SET value=value+1 WHERE key='sequence'")
            value = db.execute("SELECT value FROM counters WHERE key='sequence'").fetchone()[0]
            db.commit()
            return value

    def record_request(self, tag, request, response, error=None):
        with self._connect() as db:
            cursor = db.execute("INSERT INTO requests(tag,request_json,response_json,error,created_at) VALUES(?,?,?,?,?)",
                                (tag, json.dumps(request, ensure_ascii=False, sort_keys=True),
                                 None if response is None else json.dumps(response, ensure_ascii=False, sort_keys=True),
                                 error, self._now()))
            return cursor.lastrowid
