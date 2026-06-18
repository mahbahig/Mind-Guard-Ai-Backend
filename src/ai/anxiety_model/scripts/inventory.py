"""
Print every CSV/XLSX under the data root with guessed role (questionnaire / HRV / …).

Usage (from repo Ai_Models folder):
  python Anxiety_Model/scripts/inventory.py --root "D:\\path\\to\\unzipped_figshare"

Or set ANXIETY_DATA_DIR to that path and omit --root (defaults to Anxiety_Model/data/raw).
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_PKG not in sys.path:
    sys.path.insert(0, ROOT_PKG)


def main() -> None:
    from src.discovery import print_inventory

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=None,
        help="Unzipped dataset folder (default: ANXIETY_DATA_DIR or Anxiety_Model/data/raw)",
    )
    args = ap.parse_args()
    env = os.environ.get("ANXIETY_DATA_DIR", "").strip()
    root = args.root or env or os.path.join(ROOT_PKG, "data", "raw")
    print_inventory(root)


if __name__ == "__main__":
    main()
