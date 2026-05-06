"""SQLite conversation store — sessions, turns, context for the LLM."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

Role = Literal["user", "assistant", "system"]


@dataclass(frozen=True)
class Turn:
    id: int
    conversation_id: int
    role: Role
    content: str
    created_at: datetime


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
            CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(conversation_id, id);
            """
        )
        self._conn.commit()

    def _session_stale(self, conversation_id: int, idle: timedelta) -> bool:
        row = self._conn.execute(
            "SELECT created_at FROM turns WHERE conversation_id = ? ORDER BY id DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return False
        last = datetime.fromisoformat(row["created_at"])
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return _utc_now() - last > idle

    def get_or_create_conversation(self, idle: timedelta) -> int:
        row = self._conn.execute(
            "SELECT id FROM conversations ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is not None and not self._session_stale(row["id"], idle):
            return int(row["id"])
        now = _utc_now().isoformat()
        cur = self._conn.execute(
            "INSERT INTO conversations (started_at) VALUES (?)", (now,)
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def add_turn(self, conversation_id: int, role: Role, content: str) -> int:
        now = _utc_now().isoformat()
        cur = self._conn.execute(
            "INSERT INTO turns (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_context_messages(
        self, conversation_id: int, limit: int
    ) -> list[dict[str, str]]:
        rows = self._conn.execute(
            """
            SELECT role, content FROM turns
            WHERE conversation_id = ? AND role IN ('user', 'assistant')
            ORDER BY id DESC LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        out: list[dict[str, str]] = [
            {"role": str(r["role"]), "content": str(r["content"])} for r in reversed(rows)
        ]
        return out

    def forget_last_exchange(self, conversation_id: int) -> int:
        """Remove the most recent user + assistant pair (up to 2 rows). Returns rows deleted."""
        ids = [
            r["id"]
            for r in self._conn.execute(
                "SELECT id FROM turns WHERE conversation_id = ? ORDER BY id DESC LIMIT 2",
                (conversation_id,),
            )
        ]
        if not ids:
            return 0
        qmarks = ",".join("?" * len(ids))
        self._conn.execute(f"DELETE FROM turns WHERE id IN ({qmarks})", ids)
        self._conn.commit()
        return len(ids)
