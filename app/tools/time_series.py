from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    result = df[["timestamp", "value"]].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    return result.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def infer_frequency_seconds(df: pd.DataFrame) -> int | None:
    clean = _clean(df)
    diffs = clean["timestamp"].diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None
    mode = diffs.mode()
    if mode.empty:
        return None
    return int(mode.iloc[0].total_seconds())


def data_consistency_tool(df: pd.DataFrame) -> dict[str, Any]:
    clean = _clean(df)
    duplicate_count = int(clean["timestamp"].duplicated().sum())
    missing_values = int(clean["value"].isna().sum())
    frequency = infer_frequency_seconds(clean)

    gap_count = 0
    largest_gap_seconds = None
    irregular_steps = 0
    if frequency:
        diffs = clean["timestamp"].diff().dropna().dt.total_seconds()
        irregular_steps = int((diffs != frequency).sum())
        gap_count = int((diffs > frequency).sum())
        largest_gap_seconds = int(diffs.max()) if not diffs.empty else None

    status = "ok"
    issues: list[str] = []
    if duplicate_count:
        issues.append(f"{duplicate_count} duplicate timestamps")
    if missing_values:
        issues.append(f"{missing_values} missing values")
    if gap_count:
        issues.append(f"{gap_count} timestamp gaps")
    if irregular_steps:
        issues.append(f"{irregular_steps} irregular time steps")
    if issues:
        status = "warning"

    summary = "No major data consistency problems were found."
    if issues:
        summary = "Found " + ", ".join(issues) + "."

    return {
        "name": "data_consistency",
        "status": status,
        "summary": summary,
        "data": {
            "row_count": int(len(clean)),
            "duplicate_timestamps": duplicate_count,
            "missing_values": missing_values,
            "inferred_frequency_seconds": frequency,
            "gap_count": gap_count,
            "largest_gap_seconds": largest_gap_seconds,
            "irregular_steps": irregular_steps,
        },
    }


def _zero_runs(clean: pd.DataFrame, min_run_length: int) -> list[dict[str, Any]]:
    zero_mask = clean["value"].fillna(np.nan).eq(0)
    runs: list[dict[str, Any]] = []
    start_idx: int | None = None

    for idx, is_zero in enumerate(zero_mask):
        if is_zero and start_idx is None:
            start_idx = idx
        if (not is_zero or idx == len(zero_mask) - 1) and start_idx is not None:
            end_idx = idx if is_zero and idx == len(zero_mask) - 1 else idx - 1
            length = end_idx - start_idx + 1
            if length >= min_run_length:
                start_row = clean.iloc[start_idx]
                end_row = clean.iloc[end_idx]
                runs.append(
                    {
                        "start_timestamp": start_row["timestamp"].isoformat(),
                        "end_timestamp": end_row["timestamp"].isoformat(),
                        "length": int(length),
                    }
                )
            start_idx = None
    return runs


def _jump_outlier_candidates(
    clean: pd.DataFrame,
    threshold: float,
    top_percent: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    work = clean.dropna(subset=["value"]).copy()
    if work.empty:
        return candidates

    dt = work["timestamp"]
    months = dt.dt.month
    work["_year"] = dt.dt.year
    work["_window_id"] = np.select(
        [
            months <= 3,
            (months >= 4) & (months <= 6),
            (months >= 7) & (months <= 8),
            months >= 9,
        ],
        [1, 2, 3, 4],
    )

    for (year, window_id), group in work.groupby(["_year", "_window_id"]):
        values = group["value"].dropna()
        if len(values) < 2:
            continue

        sorted_vals = values.sort_values(ascending=False)
        top_n = max(2, int(np.ceil(len(sorted_vals) * top_percent)))
        top_vals = sorted_vals.iloc[:top_n]
        arr = top_vals.to_numpy(dtype=float)
        if len(arr) < 2:
            continue

        curr_arr = arr[:-1]
        next_arr = arr[1:]
        pct_drop = np.divide(
            curr_arr - next_arr,
            curr_arr,
            out=np.zeros_like(curr_arr, dtype=float),
            where=curr_arr != 0,
        )
        jump_positions = np.where(pct_drop > threshold)[0]
        if len(jump_positions) == 0:
            continue

        last_jump = int(jump_positions[-1])
        if last_jump + 1 >= len(arr):
            continue

        reference_value = float(arr[last_jump + 1])
        replace_indices = top_vals.index[: last_jump + 1]
        for original_index in replace_indices:
            row = work.loc[original_index]
            candidates.append(
                {
                    "timestamp": row["timestamp"].isoformat(),
                    "value": float(row["value"]),
                    "year": int(year),
                    "window_id": int(window_id),
                    "reference_value_after_jump": reference_value,
                    "drop_threshold": float(threshold),
                }
            )
    return candidates


def data_anomaly_warning_tool(
    df: pd.DataFrame,
    jump_threshold: float = 0.95,
    top_percent: float = 0.02,
    min_zero_run_length: int = 3,
) -> dict[str, Any]:
    clean = _clean(df)
    negatives = clean[clean["value"] < 0]
    negative_examples = [
        {"timestamp": row["timestamp"].isoformat(), "value": float(row["value"])}
        for _, row in negatives.head(50).iterrows()
    ]
    jump_candidates = _jump_outlier_candidates(clean, jump_threshold, top_percent)
    zero_sequences = _zero_runs(clean, min_zero_run_length)

    warning_count = len(negative_examples) + len(jump_candidates) + len(zero_sequences)
    status = "warning" if warning_count else "ok"
    summary = "No large jump values, negative values, or long zero sequences were found."
    if warning_count:
        summary = (
            f"Found {len(jump_candidates)} possible huge jump values, "
            f"{int(len(negatives))} negative values, and {len(zero_sequences)} long zero sequences."
        )

    return {
        "name": "data_anomaly_warnings",
        "status": status,
        "summary": summary,
        "data": {
            "jump_threshold": jump_threshold,
            "top_percent": top_percent,
            "min_zero_run_length": min_zero_run_length,
            "jump_candidates": jump_candidates[:50],
            "jump_candidate_count": len(jump_candidates),
            "negative_values": negative_examples,
            "negative_value_count": int(len(negatives)),
            "zero_sequences": zero_sequences[:50],
            "zero_sequence_count": len(zero_sequences),
        },
    }


def outlier_detection_tool(df: pd.DataFrame, window: int = 24, z_threshold: float = 3.5) -> dict[str, Any]:
    clean = _clean(df).dropna(subset=["value"])
    if len(clean) < max(8, window):
        return {
            "name": "outlier_detection",
            "status": "unavailable",
            "summary": "There is not enough data to check outliers reliably.",
            "data": {"outlier_count": 0, "examples": []},
        }

    rolling_median = clean["value"].rolling(window=window, min_periods=max(4, window // 3)).median()
    residual = (clean["value"] - rolling_median).abs()
    mad = residual.rolling(window=window, min_periods=max(4, window // 3)).median()
    robust_z = residual / mad.replace(0, np.nan)
    outlier_mask = robust_z > z_threshold

    if not bool(outlier_mask.any()):
        global_median = clean["value"].median()
        global_mad = (clean["value"] - global_median).abs().median()
        if global_mad == 0:
            q1 = clean["value"].quantile(0.25)
            q3 = clean["value"].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                outlier_mask = clean["value"] != global_median
            else:
                outlier_mask = (clean["value"] < q1 - 1.5 * iqr) | (clean["value"] > q3 + 1.5 * iqr)
        else:
            outlier_mask = ((clean["value"] - global_median).abs() / global_mad) > z_threshold

    outliers = clean[outlier_mask].copy()

    examples = [
        {
            "timestamp": row["timestamp"].isoformat(),
            "value": float(row["value"]),
        }
        for _, row in outliers.head(10).iterrows()
    ]
    count = int(len(outliers))
    status = "warning" if count else "ok"
    summary = "No strong outliers were found."
    if count:
        summary = f"Found {count} possible outlier points. These values are unusual compared with nearby values."

    return {
        "name": "outlier_detection",
        "status": status,
        "summary": summary,
        "data": {"outlier_count": count, "examples": examples},
    }


def historical_summary_tool(df: pd.DataFrame) -> dict[str, Any]:
    clean = _clean(df).dropna(subset=["value"])
    if clean.empty:
        return {
            "name": "historical_summary",
            "status": "unavailable",
            "summary": "There is no usable historical value data.",
            "data": {},
        }

    first_value = float(clean.iloc[0]["value"])
    last_value = float(clean.iloc[-1]["value"])
    change = last_value - first_value
    change_pct = None if first_value == 0 else (change / first_value) * 100
    peak = clean.loc[clean["value"].idxmax()]
    low = clean.loc[clean["value"].idxmin()]
    hourly = clean.assign(hour=clean["timestamp"].dt.hour).groupby("hour")["value"].mean()

    direction = "stable"
    if change_pct is not None and change_pct > 5:
        direction = "increasing"
    elif change_pct is not None and change_pct < -5:
        direction = "decreasing"

    summary = f"The historical data looks mostly {direction}. The average value is {clean['value'].mean():.2f}."
    return {
        "name": "historical_summary",
        "status": "ok",
        "summary": summary,
        "data": {
            "start": clean["timestamp"].min().isoformat(),
            "end": clean["timestamp"].max().isoformat(),
            "mean": float(clean["value"].mean()),
            "median": float(clean["value"].median()),
            "min": float(clean["value"].min()),
            "max": float(clean["value"].max()),
            "first_value": first_value,
            "last_value": last_value,
            "change": float(change),
            "change_percent": None if change_pct is None else float(change_pct),
            "peak_timestamp": peak["timestamp"].isoformat(),
            "low_timestamp": low["timestamp"].isoformat(),
            "highest_average_hour": int(hourly.idxmax()) if not hourly.empty else None,
            "lowest_average_hour": int(hourly.idxmin()) if not hourly.empty else None,
        },
    }


def hourly_consumption_context_tool(df: pd.DataFrame, days: int = 30) -> dict[str, Any]:
    clean = _clean(df).dropna(subset=["value"])
    if clean.empty:
        return {
            "name": "hourly_consumption_context",
            "status": "unavailable",
            "summary": "There is no usable historical data for hourly consumption context.",
            "data": {},
        }

    end_ts = clean["timestamp"].max()
    start_ts = end_ts - pd.Timedelta(days=days)
    recent = clean[clean["timestamp"] > start_ts].copy()
    if recent.empty:
        recent = clean.copy()

    recent["hour"] = recent["timestamp"].dt.hour
    grouped = recent.groupby("hour")["value"]
    hourly = grouped.agg(["count", "mean", "median", "min", "max", "std"]).fillna(0)
    context = [
        {
            "hour": int(hour),
            "count": int(row["count"]),
            "mean": float(row["mean"]),
            "median": float(row["median"]),
            "min": float(row["min"]),
            "max": float(row["max"]),
            "std": float(row["std"]),
        }
        for hour, row in hourly.sort_index().iterrows()
    ]

    return {
        "name": "hourly_consumption_context",
        "status": "ok",
        "summary": f"Built hourly consumption context from the last {days} historical days.",
        "data": {
            "days": days,
            "start_timestamp": recent["timestamp"].min().isoformat(),
            "end_timestamp": recent["timestamp"].max().isoformat(),
            "hourly_context": context,
        },
    }
