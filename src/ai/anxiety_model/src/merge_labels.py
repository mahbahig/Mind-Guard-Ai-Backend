"""
Build a supervised table: HRV segment rows labeled by the GAD-7 from the
questionnaire session that follows the segment (data collected before the visit).

Window: segment_time in [questionnaire_time - lookback_days, questionnaire_time).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from . import columns_util
from .discovery import find_paths_by_kind

# Baigutanova Figshare: wide survey with GAD7_1 / GAD7_2 / GAD7_F (no visit dates in file).
_WAVE_GAD_COLS = (
    ("GAD7_1", 0),
    ("GAD7_2", 14),
    ("GAD7_F", 28),
)


def _is_wide_baigutanova_survey(df: pd.DataFrame) -> bool:
    cols = {c.upper() for c in df.columns}
    return "DEVICEID" in cols and "GAD7_1" in cols and "GAD7_2" in cols and "GAD7_F" in cols


def _standardize_baigutanova_survey_wide(
    df: pd.DataFrame,
    first_sensor_time_by_subject: pd.Series,
    source: str,
) -> pd.DataFrame:
    """Synthetic questionnaire times: first HRV segment time + 0 / 14 / 28 days (biweekly design)."""
    if "deviceId" not in df.columns:
        raise ValueError(f"Expected deviceId in wide survey {source}")
    rows = []
    for _, r in df.iterrows():
        sub = str(r["deviceId"])
        if sub not in first_sensor_time_by_subject.index:
            continue
        t0 = pd.Timestamp(first_sensor_time_by_subject.loc[sub])
        for col, day_offset in _WAVE_GAD_COLS:
            if col not in df.columns:
                continue
            g = pd.to_numeric(r[col], errors="coerce")
            if pd.isna(g):
                continue
            rows.append(
                {
                    "subject": sub,
                    "q_time": t0 + pd.Timedelta(days=day_offset),
                    "gad7": float(g),
                }
            )
    if not rows:
        raise ValueError(f"No questionnaire rows after wide expansion for {source}")
    return pd.DataFrame(rows)


def _standardize_questionnaires(df: pd.DataFrame, source: str) -> pd.DataFrame:
    sid = columns_util.infer_subject_column(df)
    if sid is None:
        raise ValueError(f"Cannot infer participant/subject column in {source}. Columns: {list(df.columns)}")
    tcol = columns_util.infer_time_column(df)
    if tcol is None:
        raise ValueError(f"Cannot infer date/time column in questionnaire file {source}.")
    gad = columns_util.compute_gad7_total(df)
    out = pd.DataFrame(
        {
            "subject": df[sid].astype(str),
            "q_time": columns_util.parse_datetimes(df[tcol]),
            "gad7": gad,
        }
    )
    out = out.dropna(subset=["q_time", "gad7"])
    return out


def _standardize_hrv(df: pd.DataFrame, source: str) -> pd.DataFrame:
    sid = columns_util.infer_subject_column(df)
    if sid is None:
        raise ValueError(f"Cannot infer participant column in HRV file {source}.")
    tcol = columns_util.infer_time_column(df)
    if tcol is None:
        raise ValueError(
            f"No timestamp column found in HRV file {source}. "
            "If your export is one-row-per-participant means, use participant-level mode in train_gad7."
        )
    time = columns_util.parse_datetimes(df[tcol])
    out = df.copy()
    out["_subject"] = df[sid].astype(str)
    out["_seg_time"] = time
    out = out.dropna(subset=["_seg_time"])
    drop_cols = {"_subject", "_seg_time", sid, tcol}
    feat_cols = [c for c in out.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(out[c])]
    feat_cols = [
        c
        for c in feat_cols
        if "gad" not in c.lower() and "phq" not in c.lower() and "isi" not in c.lower()
    ]
    if not feat_cols:
        num = out.select_dtypes(include=[np.number]).columns.tolist()
        feat_cols = [
            c
            for c in num
            if c not in drop_cols and "gad" not in c.lower() and "phq" not in c.lower() and "isi" not in c.lower()
        ]
    base = out[["_subject", "_seg_time"] + feat_cols].copy()
    base.rename(columns={"_subject": "subject", "_seg_time": "seg_time"}, inplace=True)
    return base


def build_segment_table(
    data_root: str | Path,
    lookback_days: int = 14,
) -> pd.DataFrame:
    """
    Each row: one HRV segment + gad7 score from the next questionnaire that closes the window.
    """
    root = Path(data_root)
    kinds = find_paths_by_kind(root)
    q_paths = list(dict.fromkeys(kinds.get("questionnaire", []) + kinds.get("mixed", [])))
    h_paths = list(dict.fromkeys(kinds.get("hrv", []) + kinds.get("mixed", [])))

    if not q_paths:
        raise FileNotFoundError(
            "No questionnaire-like CSV found (need columns containing GAD/PHQ). "
            "Run: python scripts/inventory.py --root <path_to_unzipped_data>"
        )
    if not h_paths:
        raise FileNotFoundError(
            "No HRV-like CSV found (need SDNN/RMSSD/LF-HF style columns). "
            "Run: python scripts/inventory.py --root <path_to_unzipped_data>"
        )

    filt = [p for p in h_paths if "filtered" in p.name.lower()]
    if filt:
        h_paths = filt

    h_frames = []
    for p in h_paths:
        df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
        h_frames.append(_standardize_hrv(df, str(p)))
    hrv = pd.concat(h_frames, ignore_index=True)
    first_sensor_time = hrv.groupby("subject")["seg_time"].min()

    q_frames = []
    for p in q_paths:
        df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
        if _is_wide_baigutanova_survey(df):
            q_frames.append(_standardize_baigutanova_survey_wide(df, first_sensor_time, str(p)))
        else:
            q_frames.append(_standardize_questionnaires(df, str(p)))
    quest = pd.concat(q_frames, ignore_index=True).sort_values(["subject", "q_time"])

    look = pd.Timedelta(days=lookback_days)
    labeled_parts = []
    for (sub, gdf) in quest.groupby("subject"):
        hsub = hrv[hrv["subject"] == sub]
        if hsub.empty:
            continue
        for _, qrow in gdf.iterrows():
            q_time = qrow["q_time"]
            win_start = q_time - look
            mask = (hsub["seg_time"] >= win_start) & (hsub["seg_time"] < q_time)
            block = hsub.loc[mask].copy()
            if block.empty:
                continue
            block["gad7"] = float(qrow["gad7"])
            block["q_time"] = q_time
            labeled_parts.append(block)

    if not labeled_parts:
        return pd.DataFrame()

    return pd.concat(labeled_parts, ignore_index=True)


def build_participant_mean_table(data_root: str | Path) -> pd.DataFrame:
    """
    Fallback: one row per participant — mean of numeric HRV columns over all segments,
    merged with the latest non-missing GAD-7 for that participant.
    """
    root = Path(data_root)
    kinds = find_paths_by_kind(root)
    q_paths = list(dict.fromkeys(kinds.get("questionnaire", []) + kinds.get("mixed", [])))
    h_paths = list(dict.fromkeys(kinds.get("hrv", []) + kinds.get("mixed", [])))
    if not q_paths or not h_paths:
        raise FileNotFoundError("Need at least one questionnaire file and one HRV file.")

    filt = [p for p in h_paths if "filtered" in p.name.lower()]
    if filt:
        h_paths = filt

    h_blocks = []
    for p in h_paths:
        df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
        h_blocks.append(_standardize_hrv(df, str(p)))
    hrv = pd.concat(h_blocks, ignore_index=True)
    first_sensor_time = hrv.groupby("subject")["seg_time"].min()

    quest_parts = []
    for p in q_paths:
        df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
        if _is_wide_baigutanova_survey(df):
            quest_parts.append(_standardize_baigutanova_survey_wide(df, first_sensor_time, str(p)))
        else:
            quest_parts.append(_standardize_questionnaires(df, str(p)))
    quest = pd.concat(quest_parts, ignore_index=True)
    quest = quest.sort_values(["subject", "q_time"]).dropna(subset=["gad7"])
    quest_last = quest.groupby("subject", as_index=False).last()

    blocks: list[pd.DataFrame] = []
    for p in h_paths:
        df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
        sid = columns_util.infer_subject_column(df)
        if sid is None:
            continue
        num = df.select_dtypes(include=[np.number])
        drop_lab = [c for c in num.columns if any(x in c.lower() for x in ("gad", "phq", "isi"))]
        num = num.drop(columns=drop_lab, errors="ignore")
        if num.shape[1] == 0:
            continue
        block = pd.concat([df[sid].astype(str).rename("subject"), num.reset_index(drop=True)], axis=1)
        blocks.append(block)
    if not blocks:
        raise ValueError("Could not aggregate any numeric HRV columns.")
    hall = pd.concat(blocks, ignore_index=True)
    hmean = hall.groupby("subject", as_index=False).mean(numeric_only=True)

    merged = quest_last.merge(hmean, on="subject", how="inner")
    return merged
