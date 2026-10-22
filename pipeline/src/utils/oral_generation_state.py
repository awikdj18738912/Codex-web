"""Durable attempt IDs and completed records for continuous Oral generation."""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
from pathlib import Path


class OralGenerationState:
    """SQLite is authoritative; JSONL and the legacy checkpoint are readable mirrors.

    A result is committed before it is appended to JSONL. On restart, rebuilding
    that mirror repairs interruptions between those two writes. Reserved but
    unfinished IDs are retried, so out-of-order completion cannot duplicate IDs.
    """

    def __init__(self, directory: Path, *, resume: bool = True):
        directory.mkdir(parents=True, exist_ok=True)
        self.output_path = directory / "oral.jsonl"
        self.checkpoint_path = directory / "oral.ckpt.json"
        self._lock = (directory / "oral.generation.lock").open("a")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RuntimeError(f"Oral generation already owns {directory}") from None
        self.db = sqlite3.connect(directory / "oral.state.sqlite3")
        try:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    record_json TEXT,
                    completion_order INTEGER
                );
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
            """)
            if not resume:
                with self.db:
                    self.db.execute("DELETE FROM attempts")
                    self.db.execute("DELETE FROM state")
                self.output_path.unlink(missing_ok=True)
                self.checkpoint_path.unlink(missing_ok=True)
            if self.db.execute("SELECT value FROM state WHERE key='next_id'").fetchone() is None:
                self._migrate_legacy()
            self.next_id = self.db.execute("SELECT value FROM state WHERE key='next_id'").fetchone()[0]
            self.generated = self.db.execute("SELECT COUNT(*) FROM attempts WHERE status='valid'").fetchone()[0]
            self.completed = self.db.execute("SELECT COUNT(*) FROM attempts WHERE status!='pending'").fetchone()[0]
            self._rebuild_output()
            self._save_checkpoint()
        except BaseException:
            self.close()
            raise

    def _migrate_legacy(self):
        checkpoint = json.loads(self.checkpoint_path.read_text()) if self.checkpoint_path.exists() else {}
        rows = []
        ids = set()
        if self.output_path.exists():
            with self.output_path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    attempt_id = item.get("id")
                    if type(attempt_id) is not int or attempt_id < 0 or attempt_id in ids:
                        raise ValueError("Legacy Oral records require unique nonnegative integer IDs")
                    ids.add(attempt_id)
                    rows.append(item)
        next_id = max(int(checkpoint.get("attempted_count", checkpoint.get("source_offset", 0))),
                      max(ids, default=-1) + 1)
        with self.db:
            self.db.executemany("INSERT INTO attempts(attempt_id,status) VALUES (?,'invalid')",
                                ((i,) for i in range(next_id)))
            self.db.executemany(
                "UPDATE attempts SET status='valid',record_json=?,completion_order=? WHERE attempt_id=?",
                ((json.dumps(item, ensure_ascii=False), order, item["id"])
                 for order, item in enumerate(rows)),
            )
            self.db.executemany("INSERT INTO state(key,value) VALUES (?,?)",
                                [("next_id", next_id), ("next_completion", len(rows))])

    def _rebuild_output(self):
        temporary = self.output_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for (record,) in self.db.execute(
                "SELECT record_json FROM attempts WHERE status='valid' ORDER BY completion_order"
            ):
                stream.write(record + "\n")
        os.replace(temporary, self.output_path)

    def _save_checkpoint(self):
        temporary = self.checkpoint_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "generated_count": self.generated,
            "attempted_count": self.next_id,
            "completed_attempts": self.completed,
            "pending_attempts": self.next_id - self.completed,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.checkpoint_path)

    def pending_ids(self) -> list[int]:
        return [row[0] for row in self.db.execute(
            "SELECT attempt_id FROM attempts WHERE status='pending' ORDER BY attempt_id"
        )]

    def reserve(self) -> int:
        attempt_id = self.next_id
        with self.db:
            self.db.execute("INSERT INTO attempts(attempt_id,status) VALUES (?,'pending')", (attempt_id,))
            self.db.execute("UPDATE state SET value=? WHERE key='next_id'", (attempt_id + 1,))
        self.next_id += 1
        return attempt_id

    def finish(self, attempt_id: int, record: dict | None):
        if record is not None and record.get("id") != attempt_id:
            raise ValueError("Generated record ID differs from reserved attempt ID")
        serialized = json.dumps(record, ensure_ascii=False) if record is not None else None
        with self.db:
            order = self.db.execute("SELECT value FROM state WHERE key='next_completion'").fetchone()[0]
            result = self.db.execute(
                "UPDATE attempts SET status=?,record_json=?,completion_order=? "
                "WHERE attempt_id=? AND status='pending'",
                ("valid" if record is not None else "invalid", serialized,
                 order if record is not None else None, attempt_id),
            )
            if result.rowcount != 1:
                raise ValueError(f"Attempt {attempt_id} is not pending")
            if record is not None:
                self.db.execute("UPDATE state SET value=? WHERE key='next_completion'", (order + 1,))
        self.completed += 1
        if record is not None:
            self.generated += 1
            self._append(serialized)
        self._save_checkpoint()

    def _append(self, serialized: str):
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(serialized + "\n")

    def close(self):
        self.db.close()
        self._lock.close()
