"""Infer common column names from messy real-world CSV headers."""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd


def _norm(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())


def infer_subject_column(df: pd.DataFrame) -> Optional[str]:
    priority = [
        "deviceid",
        "device_id",
        "participant_id",
        "subject_id",
        "user_id",
        "pid",
        "id",
    ]
    lowered = {_norm(c): c for c in df.columns}
    for p in priority:
        if p in lowered:
            return lowered[p]
    for c in df.columns:
        n = _norm(c)
        if any(k in n for k in ("participant", "subject", "device", "user")) and "item" not in n:
            return c
    return None


def infer_time_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        n = _norm(c)
        if any(
            k in n
            for k in (
                "timestamp",
                "datetime",
                "segment_start",
                "segment_end",
                "start_time",
                "time",
                "date",
                "day",
            )
        ):
            if df[c].dtype == object or "datetime" in str(df[c].dtype):
                return c
    for c in df.columns:
        if "time" in _norm(c) or _norm(c) == "date":
            return c
    return None


def infer_gad_total_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        n = _norm(c)
        if "gad" not in n:
            continue
        if any(x in n for x in ("item", "question", "q1", "q2", "gad1", "gad2")):
            continue
        if any(x in n for x in ("total", "score", "sum", "gad7", "gad_7", "gad-7")):
            return c
    for c in df.columns:
        n = _norm(c)
        if n in ("gad7", "gad_7", "gad", "gad_total"):
            return c
    return None


def infer_gad_item_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        n = _norm(c)
        if re.match(r"^gad[_\-]?[0-9]+$", n) or re.match(r"^gad7[_\-]?[0-9]+$", n):
            cols.append(c)
    return sorted(cols, key=lambda x: _norm(x))


def compute_gad7_total(df: pd.DataFrame) -> pd.Series:
    total_col = infer_gad_total_column(df)
    if total_col:
        return pd.to_numeric(df[total_col], errors="coerce")
    items = infer_gad_item_columns(df)
    if len(items) >= 7:
        sub = df[items[:7]].apply(pd.to_numeric, errors="coerce")
        return sub.sum(axis=1, min_count=7)
    raise ValueError("Could not find GAD-7 total or 7 item columns.")


def parse_datetimes(series: pd.Series) -> pd.Series:
    """
    Parse strings or datetimes. If values look like Unix epoch **milliseconds**
    (common in Samsung export), use unit='ms' — plain to_datetime(int) is wrong.
    """
    if pd.api.types.is_numeric_dtype(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if not s.empty and float(s.median()) > 1e11:
            dt = pd.to_datetime(series, unit="ms", errors="coerce")
            if getattr(dt.dtype, "tz", None) is not None:
                return dt.dt.tz_convert(None)
            return dt
    dt = pd.to_datetime(series, errors="coerce")
    if getattr(dt.dtype, "tz", None) is not None:
        return dt.dt.tz_convert(None)
    return dt
