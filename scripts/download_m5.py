"""Download the M5 retail dataset (Kaggle) into data/raw/.

Usage:
    python scripts/download_m5.py

Dataset: "M5 Forecasting - Accuracy" (Makridakis et al., 2022)
https://www.kaggle.com/competitions/m5-forecasting-accuracy

Files pulled (per Docs section 4.1-4.2):
    sales_train_evaluation.csv  - daily unit sales by SKU-store
    sell_prices.csv             - weekly selling price by item-store
    calendar.csv                - dates, events, SNAP flags
    sales_train_validation.csv  - (used by some calibration recipes)
    sample_submission.csv       - (ignored by the pipeline, downloaded anyway)

Authentication: kagglehub reads ~/.kaggle/kaggle.json. Run
`kaggle competitions download` once interactively, or place your API token
at ~/.kaggle/kaggle.json (Kaggle -> Settings -> Create New Token).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

# Primary: public dataset mirror (anonymous download, no Kaggle account or
# competition-rule acceptance needed). Fallback: official competition route.
KAGGLE_DATASET = "aryayadav0513/m5-forecasting-accuracy"
KAGGLE_COMPETITION = "m5-forecasting-accuracy"

REQUIRED_FILES = [
    "sales_train_evaluation.csv",
    "sell_prices.csv",
    "calendar.csv",
    "sales_train_validation.csv",
]


def download_m5(dest: Path = RAW_DIR) -> list[Path]:
    """Download M5 via kagglehub and place required CSVs in dest."""
    import kagglehub  # deferred so the rest of the script can print help without it

    dest.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading dataset {KAGGLE_DATASET} via kagglehub (anonymous) ...")
        cached_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:
        print(f"Mirror download failed ({exc}); trying official competition route ...")
        print(f"Downloading competition {KAGGLE_COMPETITION} via kagglehub ...")
        cached_dir = Path(kagglehub.competition_download(KAGGLE_COMPETITION))
    print(f"Cached at: {cached_dir}")

    copied: list[Path] = []
    for name in REQUIRED_FILES:
        matches = sorted(cached_dir.rglob(name))  # mirrors may nest files in subfolders
        if not matches:
            print(f"  WARNING: expected file missing in download: {name}")
            continue
        src = matches[0]
        out = dest / name
        if out.exists():
            print(f"  exists, skipping: {out}")
        else:
            print(f"  copying: {name} ({src.stat().st_size / 1e6:.1f} MB)")
            shutil.copy2(src, out)
        copied.append(out)

    return copied


def main() -> int:
    print(f"Repo root : {REPO_ROOT}")
    print(f"Target dir: {RAW_DIR}")

    if all((RAW_DIR / f).exists() for f in REQUIRED_FILES):
        print("All M5 files already present - nothing to do.")
        return 0

    try:
        copied = download_m5()
    except Exception as exc:  # noqa: BLE001 - report cleanly for the user
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        print(
            "\nFix options:\n"
            "  1. Create a Kaggle API token (kaggle.com -> Settings -> Create New Token)\n"
            "     and save it as ~/.kaggle/kaggle.json, then re-run this script.\n"
            "  2. Manually download the competition files and place them in\n"
            f"     {RAW_DIR}:\n"
            "       sales_train_evaluation.csv, sell_prices.csv, calendar.csv,\n"
            "       sales_train_validation.csv\n",
            file=sys.stderr,
        )
        return 1

    missing = [f for f in REQUIRED_FILES if not (RAW_DIR / f).exists()]
    if missing:
        print(f"FAILED - missing files after download: {missing}", file=sys.stderr)
        return 1

    print(f"\nDone. {len(copied)} files ready in {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
