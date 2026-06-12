from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from app.config import get_settings
from app.database import get_connection
from app.schemas import DatasetSummary
from app.services.chat_store import utc_now
from app.tools.time_series import infer_frequency_seconds


REQUIRED_COLUMNS = {"timestamp", "value"}


def _validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns: {sorted(REQUIRED_COLUMNS)}")
    clean = df[["timestamp", "value"]].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce")
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    if clean["timestamp"].isna().all():
        raise ValueError("CSV timestamp column could not be parsed.")
    clean = clean.sort_values("timestamp").reset_index(drop=True)
    return clean


def save_uploaded_dataset(file: BinaryIO, filename: str) -> DatasetSummary:
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = str(uuid.uuid4())
    suffix = Path(filename).suffix or ".csv"
    storage_path = settings.upload_dir / f"{dataset_id}{suffix}"

    with storage_path.open("wb") as out:
        shutil.copyfileobj(file, out)

    try:
        df = _validate_frame(pd.read_csv(storage_path))
    except Exception:
        storage_path.unlink(missing_ok=True)
        raise

    df.to_csv(storage_path, index=False)
    missing_values = int(df["value"].isna().sum())
    valid_timestamps = df["timestamp"].dropna()
    start = valid_timestamps.min().isoformat() if not valid_timestamps.empty else None
    end = valid_timestamps.max().isoformat() if not valid_timestamps.empty else None
    frequency = infer_frequency_seconds(df)

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO datasets (
                id, filename, storage_path, row_count, start_timestamp,
                end_timestamp, inferred_frequency_seconds, missing_values, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                filename,
                str(storage_path),
                int(len(df)),
                start,
                end,
                frequency,
                missing_values,
                utc_now(),
            ),
        )

    return DatasetSummary(
        dataset_id=dataset_id,
        filename=filename,
        row_count=int(len(df)),
        start_timestamp=start,
        end_timestamp=end,
        inferred_frequency_seconds=frequency,
        missing_values=missing_values,
    )


def get_dataset(dataset_id: str) -> DatasetSummary | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    if row is None:
        return None
    return DatasetSummary(
        dataset_id=row["id"],
        filename=row["filename"],
        row_count=row["row_count"],
        start_timestamp=row["start_timestamp"],
        end_timestamp=row["end_timestamp"],
        inferred_frequency_seconds=row["inferred_frequency_seconds"],
        missing_values=row["missing_values"],
    )


def get_dataset_path(dataset_id: str) -> Path:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT storage_path FROM datasets WHERE id = ?",
            (dataset_id,),
        ).fetchone()
    if row is None:
        raise FileNotFoundError(f"Dataset not found: {dataset_id}")
    return Path(row["storage_path"])


def load_dataset_frame(dataset_id: str) -> pd.DataFrame:
    path = get_dataset_path(dataset_id)
    return _validate_frame(pd.read_csv(path))
