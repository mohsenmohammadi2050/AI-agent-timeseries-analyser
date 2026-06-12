from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from app.database import get_connection


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_chat(title: str | None = None) -> str:
    thread_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chats (id, title, created_at) VALUES (?, ?, ?)",
            (thread_id, title, utc_now()),
        )
    return thread_id


def ensure_chat(thread_id: str) -> None:
    with get_connection() as conn:
        row = conn.execute("SELECT id FROM chats WHERE id = ?", (thread_id,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO chats (id, title, created_at) VALUES (?, ?, ?)",
                (thread_id, None, utc_now()),
            )


def add_message(
    thread_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    ensure_chat(thread_id)
    message_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO messages (id, thread_id, role, content, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                thread_id,
                role,
                content,
                json.dumps(metadata or {}),
                utc_now(),
            ),
        )
    return message_id


def get_messages(thread_id: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content, metadata_json, created_at
            FROM messages
            WHERE thread_id = ?
            ORDER BY created_at ASC
            """,
            (thread_id,),
        ).fetchall()
    return [
        {
            "role": row["role"],
            "content": row["content"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def add_agent_run(
    thread_id: str,
    mode: str,
    input_payload: dict[str, Any],
    output_payload: dict[str, Any],
) -> str:
    run_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_runs (id, thread_id, mode, input_json, output_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                thread_id,
                mode,
                json.dumps(input_payload, default=str),
                json.dumps(output_payload, default=str),
                utc_now(),
            ),
        )
    return run_id
