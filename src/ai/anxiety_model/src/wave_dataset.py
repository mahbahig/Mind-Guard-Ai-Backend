"""
Build person x wave (or person-level) aggregated training rows (Baigutanova plan).

Wave windows (per subject, anchored at first HRV segment time t0):
  Wave 1: [t0, t0 + 14d)           -> GAD7_1
  Wave 2: [t0 + 14d, t0 + 28d)     -> GAD7_2
  Wave F: [t0 + 28d, max_ts + 1ns) -> GAD7_F

Enhanced features (plan): drop redundant raw columns, richer stats, day/night HRV,
demographics, segment compliance. Person-level mode collapses waves per subject.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

# Raw segment columns to exclude before aggregation (multicollinear / redundant).
_DROP_SOURCE_COLS = frozenset(
    {
        "lf",  # keep hf; lf/hf ratio kept if present as lf/hf
        "pnn20",  # keep pnn50
        "sdsd",  # keep rmssd
        "distance",
        "calories",
    }
)

# HRV-related columns get full statistic set (mean, std, median, min, max, iqr, cv, count, slope).
_RICH_STATS_PREFIXES: tuple[str, ...] = (
    "missingness_score",
    "HR",
    "ibi",
    "sdnn",
    "rmssd",
    "pnn50",
    "hf",
    "lf/hf",
)

# Day vs night (8:00–19:59 daytime) aggregates for these base column names.
_DAYNIGHT_BASE_COLS = ("HR", "ibi", "sdnn", "rmssd", "lf/hf")

# Survey columns merged as static subject features (numeric / ordinal in CSV).
_DEMO_COLS = (
    "sex",
    "age",
    "height",
    "weight",
    "marriage",
    "occupation",
    "smartwatch",
    "regular",
    "exercise",
    "coffee",
    "smoking",
    "drinking",
)


def _load_hrv_filtered(root: Path) -> pd.DataFrame:
    p = root / "sensor_hrv_filtered.csv"
    if not p.is_file():
        p = root / "sensor_hrv.csv"
    if not p.is_file():
        raise FileNotFoundError(f"No sensor_hrv_filtered.csv or sensor_hrv.csv under {root}")
    return pd.read_csv(p)


def _hrv_numeric_columns(df: pd.DataFrame) -> list[str]:
    skip = {"deviceId", "ts_start", "ts_end", "subject", "seg_time"}
    cols = []
    for c in df.columns:
        if c in skip:
            continue
        if c in _DROP_SOURCE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if any(x in c.lower() for x in ("gad", "phq", "isi")):
                continue
            cols.append(c)
    return cols


def _is_rich_stats_col(col_name: str) -> bool:
    if col_name in _RICH_STATS_PREFIXES:
        return True
    return any(col_name.startswith(p) for p in _RICH_STATS_PREFIXES if p not in ("lf/hf",))


def _agg_series_stats(s: pd.Series, *, rich: bool) -> dict[str, float]:
    """Aggregate one numeric segment series to flat feature dict."""
    v = pd.to_numeric(s, errors="coerce").dropna()
    out: dict[str, float] = {}
    if len(v) == 0:
        keys = ["mean", "std", "median", "min", "max", "iqr", "cv", "count", "slope"] if rich else ["mean", "std", "count"]
        for k in keys:
            out[k] = float("nan")
        return out

    arr = v.to_numpy(dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0)) if n > 1 else 0.0
    out["mean"] = mean
    out["std"] = std
    out["count"] = float(n)
    if not rich:
        return out

    out["median"] = float(np.median(arr))
    out["min"] = float(np.min(arr))
    out["max"] = float(np.max(arr))
    q25, q75 = np.percentile(arr, [25, 75])
    out["iqr"] = float(q75 - q25)
    out["cv"] = float(std / abs(mean)) if abs(mean) > 1e-12 else float("nan")
    if n > 2:
        t = np.arange(n, dtype=float)
        out["slope"] = float(np.polyfit(t, arr, 1)[0])
    else:
        out["slope"] = float("nan")
    return out


def _flatten_stats(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    safe = prefix.replace("/", "_per_")
    return {f"{safe}_{k}": float(stats[k]) for k in stats}


def _daynight_masks(hours: pd.Series) -> tuple[pd.Series, pd.Series]:
    h = hours.astype(int)
    day = (h >= 8) & (h <= 19)
    night = ~day
    return day, night


def _aggregate_day_night(
    block: pd.DataFrame,
    col: str,
    hours: pd.Series,
) -> dict[str, float]:
    if col not in block.columns:
        return {}
    s_all = pd.to_numeric(block[col], errors="coerce")
    day, night = _daynight_masks(hours)
    rec: dict[str, float] = {}
    for name, m in (("day", day), ("night", night)):
        sb = s_all.loc[m].dropna()
        rec[f"{col.replace('/', '_per_')}_{name}_mean"] = float(sb.mean()) if len(sb) else float("nan")
        rec[f"{col.replace('/', '_per_')}_{name}_std"] = float(sb.std(ddof=0)) if len(sb) > 1 else float("nan")
    dmean = rec.get(f"{col.replace('/', '_per_')}_day_mean", float("nan"))
    nmean = rec.get(f"{col.replace('/', '_per_')}_night_mean", float("nan"))
    if np.isfinite(dmean) and np.isfinite(nmean):
        rec[f"{col.replace('/', '_per_')}_day_minus_night_mean"] = float(dmean - nmean)
    return rec


def _survey_demographics_row(srow: pd.Series) -> dict[str, float]:
    rec: dict[str, float] = {}
    for c in _DEMO_COLS:
        if c not in srow.index:
            continue
        v = pd.to_numeric(srow.get(c), errors="coerce")
        rec[f"demo_{c}"] = float(v) if pd.notna(v) else float("nan")
    h = pd.to_numeric(srow.get("height"), errors="coerce")
    w = pd.to_numeric(srow.get("weight"), errors="coerce")
    if pd.notna(h) and pd.notna(w) and float(h) > 1e-6:
        rec["demo_bmi"] = float(w) / ((float(h) / 100.0) ** 2)
    else:
        rec["demo_bmi"] = float("nan")
    return rec


def build_person_wave_table(
    data_root: str | Path,
    include_sleep: bool = True,
) -> pd.DataFrame:
    """
    One row per (deviceId, wave) with enhanced aggregates, gad7 label, optional sleep diary.
    """
    root = Path(data_root)
    hrv_raw = _load_hrv_filtered(root)
    hrv = hrv_raw.copy()
    hrv["seg_time"] = pd.to_datetime(hrv["ts_start"], unit="ms", utc=True).dt.tz_convert(None)

    num_cols = _hrv_numeric_columns(hrv)
    if not num_cols:
        raise ValueError("No numeric HRV feature columns found.")

    survey = pd.read_csv(root / "survey.csv")
    if not _is_wide_survey(survey):
        raise ValueError("Expected Baigutanova survey.csv with GAD7_1, GAD7_2, GAD7_F")

    sleep: Optional[pd.DataFrame] = None
    if include_sleep and (root / "sleep_diary.csv").is_file():
        sleep = pd.read_csv(root / "sleep_diary.csv")
        sleep["diary_date"] = pd.to_datetime(sleep["date"], errors="coerce").dt.normalize()

    rows: list[dict] = []
    for sub, g in hrv.groupby("deviceId"):
        g = g.sort_values("seg_time")
        t0 = g["seg_time"].min()
        t1 = t0 + pd.Timedelta(days=14)
        t2 = t0 + pd.Timedelta(days=28)
        t_end = g["seg_time"].max() + pd.Timedelta(seconds=1)

        wave_specs = [
            (1, t0, t1, "GAD7_1"),
            (2, t1, t2, "GAD7_2"),
            (3, t2, t_end, "GAD7_F"),
        ]
        srow = survey.loc[survey["deviceId"].astype(str) == str(sub)]
        if srow.empty:
            continue
        srow = srow.iloc[0]

        for wave_id, w_start, w_end, gad_col in wave_specs:
            mask = (g["seg_time"] >= w_start) & (g["seg_time"] < w_end)
            block = g.loc[mask]
            if block.empty:
                continue
            gad = pd.to_numeric(srow.get(gad_col), errors="coerce")
            if pd.isna(gad):
                continue

            rec: dict[str, object] = {"subject": str(sub), "wave": wave_id, "gad7": float(gad)}
            hours = block["seg_time"].dt.hour

            rec["n_segments"] = float(len(block))
            rec["window_span_days"] = float(max(0.0, (block["seg_time"].max() - block["seg_time"].min()).total_seconds() / 86400.0))

            for c in num_cols:
                s = pd.to_numeric(block[c], errors="coerce")
                rich = _is_rich_stats_col(c)
                st = _agg_series_stats(s, rich=rich)
                rec.update(_flatten_stats(c, st))

            for base in _DAYNIGHT_BASE_COLS:
                if base == "lf/hf" and base not in block.columns:
                    continue
                rec.update(_aggregate_day_night(block, base, hours))

            rec.update(_survey_demographics_row(srow))

            if sleep is not None:
                sd = sleep[sleep["userId"].astype(str) == str(sub)]
                if wave_id < 3:
                    sm = (sd["diary_date"] >= pd.Timestamp(w_start.date())) & (
                        sd["diary_date"] < pd.Timestamp(w_end.date())
                    )
                else:
                    sm = sd["diary_date"] >= pd.Timestamp(w_start.date())
                sdw = sd.loc[sm]
                for col in [
                    "sleep_duration",
                    "sleep_efficiency",
                    "sleep_latency",
                    "waso",
                    "wakeup@night",
                    "in_bed_duration",
                ]:
                    if col not in sdw.columns:
                        continue
                    vv = pd.to_numeric(sdw[col], errors="coerce")
                    key = "sleep_" + col.replace("@", "_") + "_mean"
                    key_std = "sleep_" + col.replace("@", "_") + "_std"
                    rec[key] = float(vv.mean()) if len(vv) else float("nan")
                    rec[key_std] = float(vv.std(ddof=0)) if len(vv) > 1 else float("nan")

            rows.append(rec)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


PersonLabelMode = Literal["mean", "max"]
PersonBinaryMode = Literal["any_wave_ge5", "max_score_ge5"]


def build_person_level_table(
    wave_df: pd.DataFrame,
    *,
    label_mode: PersonLabelMode = "mean",
    binary_mode: PersonBinaryMode = "any_wave_ge5",
) -> pd.DataFrame:
    """
    Collapse person-wave rows to one row per subject.

    - label_mode ``mean`` / ``max``: continuous GAD-7 target across waves.
    - ``binary_any_wave_ge5`` column: 1 if any wave had GAD-7 >= 5 (screening target).
    """
    if wave_df.empty:
        return pd.DataFrame()
    meta = {"subject", "wave", "gad7"}
    feat_cols = [
        c
        for c in wave_df.columns
        if c not in meta and pd.api.types.is_numeric_dtype(wave_df[c]) and not str(c).startswith("demo_")
    ]

    rows: list[dict] = []
    for sub, g in wave_df.groupby("subject"):
        g = g.sort_values("wave")
        rec: dict[str, float] = {"subject": str(sub)}
        gad_vals = g["gad7"].astype(float)
        if label_mode == "max":
            rec["gad7"] = float(gad_vals.max())
        else:
            rec["gad7"] = float(gad_vals.mean())

        for c in feat_cols:
            v = pd.to_numeric(g[c], errors="coerce")
            rec[f"person_{c}_mean"] = float(v.mean())
            rec[f"person_{c}_std"] = float(v.std(ddof=0)) if len(v.dropna()) > 1 else float("nan")
            rec[f"person_{c}_min"] = float(v.min()) if len(v.dropna()) else float("nan")
            rec[f"person_{c}_max"] = float(v.max()) if len(v.dropna()) else float("nan")

        rec["person_n_waves"] = float(len(g))
        g0 = g.iloc[0]
        for c in wave_df.columns:
            if str(c).startswith("demo_") and pd.api.types.is_numeric_dtype(wave_df[c]):
                dv = pd.to_numeric(g0.get(c), errors="coerce")
                rec[str(c)] = float(dv) if pd.notna(dv) else float("nan")
        rows.append(rec)

    out = pd.DataFrame(rows)
    any_pos = wave_df.groupby("subject")["gad7"].apply(lambda s: (s.astype(float) >= 5.0).any())
    out["binary_any_wave_ge5"] = out["subject"].map(any_pos).astype(int)
    if binary_mode == "max_score_ge5":
        mx = wave_df.groupby("subject")["gad7"].max()
        out["binary_max_score_ge5"] = (mx >= 5.0).reindex(out["subject"]).astype(int).values
    return out


def _is_wide_survey(df: pd.DataFrame) -> bool:
    cols = {c.upper() for c in df.columns}
    return "GAD7_1" in cols and "GAD7_2" in cols and "GAD7_F" in cols
