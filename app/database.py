from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.config import get_settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    start_timestamp TEXT,
    end_timestamp TEXT,
    inferred_frequency_seconds INTEGER,
    missing_values INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(thread_id) REFERENCES chats(id)
);

CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    input_json TEXT NOT NULL,
    output_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(thread_id) REFERENCES chats(id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    dataset_id TEXT,
    model_name TEXT,
    start_timestamp TEXT,
    end_timestamp TEXT,
    output_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id)
);

"""


def init_db() -> None:
    settings = get_settings()
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.database_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    settings = get_settings()
    init_db()
    conn = sqlite3.connect(Path(settings.database_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
