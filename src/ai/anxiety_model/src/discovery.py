"""
Locate Baigutanova / Figshare CSV-style tables under a user-provided root.
Uses column-name heuristics only (no hard-coded Figshare filenames).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class FileSniff:
    path: Path
    columns: list[str]
    n_rows_sample: int


def _read_columns(path: Path, nrows: int = 5) -> tuple[list[str], int]:
    if path.suffix.lower() in {".csv"}:
        df = pd.read_csv(path, nrows=nrows)
        return list(df.columns), len(df)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, nrows=nrows, engine="openpyxl" if path.suffix.lower() == ".xlsx" else None)
        return list(df.columns), len(df)
    return [], 0


def iter_tabular_files(root: Path) -> Iterable[Path]:
    root = root.resolve()
    if not root.is_dir():
        return
    for pat in ("**/*.csv", "**/*.xlsx", "**/*.xls"):
        for p in root.glob(pat):
            if p.name.startswith("~$"):
                continue
            yield p


def sniff_all(root: str | Path) -> list[FileSniff]:
    root = Path(root)
    out: list[FileSniff] = []
    for p in sorted(iter_tabular_files(root), key=lambda x: str(x).lower()):
        cols, n = _read_columns(p)
        if cols:
            out.append(FileSniff(path=p, columns=cols, n_rows_sample=n))
    return out


def _cols_lower(cols: list[str]) -> list[str]:
    return [c.lower() for c in cols]


def classify_file(sniff: FileSniff) -> str:
    lc = _cols_lower(sniff.columns)
    joined = " ".join(lc)
    has_gad = any("gad" in c for c in lc)
    has_phq = any("phq" in c for c in lc)
    has_isi = any(c.startswith("isi") or "_isi" in c or " isi" in joined for c in lc)
    hrv_markers = ("sdnn", "rmssd", "lf_hf", "lf/hf", "nn50", "pnn50")
    has_hrv = any(m in joined for m in hrv_markers) or "heart rate variability" in joined
    has_sleep_diary = any(x in joined for x in ("waso", "sleep diary", "bedtime", "wake_time", "sleeponset"))

    if has_hrv and (has_gad or has_phq or has_isi):
        return "mixed"
    if (has_gad or has_phq or has_isi) and not has_hrv:
        return "questionnaire"
    if has_hrv:
        return "hrv"
    if has_sleep_diary:
        return "sleep_diary"
    return "other"


def print_inventory(root: str | Path) -> None:
    root = Path(root)
    rows = sniff_all(root)
    if not rows:
        print(f"No .csv/.xlsx found under: {root.resolve()}")
        print("Unzip the Figshare bundle here or set ANXIETY_DATA_DIR to the folder that contains the CSVs.")
        return

    print(f"Root: {root.resolve()}\nFound {len(rows)} tabular files.\n")
    for s in rows:
        cat = classify_file(s)
        try:
            rel = s.path.relative_to(root.resolve())
        except ValueError:
            rel = s.path
        print(f"[{cat}] {rel}")
        print(f"    columns ({len(s.columns)}): {', '.join(s.columns[:18])}{' …' if len(s.columns) > 18 else ''}")
        print()


def find_paths_by_kind(root: str | Path) -> dict[str, list[Path]]:
    buckets: dict[str, list[Path]] = {
        "questionnaire": [],
        "hrv": [],
        "sleep_diary": [],
        "mixed": [],
        "other": [],
    }
    for s in sniff_all(root):
        k = classify_file(s)
        buckets.setdefault(k, []).append(s.path)
    return buckets
