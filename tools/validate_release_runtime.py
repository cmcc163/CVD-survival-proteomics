"""Validate every released database and model checkpoint at runtime.

Run this utility only on trusted artifacts. Python pickle and PyTorch checkpoint
formats can execute code while loading; the repository's release importer first
performs a path/privacy scan without deserializing them.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sqlite3
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


OUTCOMES = ("ASCVD", "HF", "Total_CVD")
ALGORITHMS = (
    "LinearModel",
    "XGBoost",
    "MLP",
    "TabNet",
    "NODE",
    "FT-Transformer",
    "SAINT",
    "TabPFN",
)
PREDICTOR_SETS = ("classic", "protein_lassonet", "all_lassonet")
SUFFIXES = (".pt", ".pkl", ".pth", ".joblib")


def locate_fold(root: Path, outcome: str, algorithm: str, predictors: str, fold: int) -> Path:
    directory = root / "models" / outcome / algorithm / predictors
    matches = [directory / f"fold_{fold}{suffix}" for suffix in SUFFIXES]
    existing = [path for path in matches if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(f"Expected exactly one checkpoint for {directory}/fold_{fold}")
    return existing[0]


def validate_database(path: Path) -> int:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        studies = connection.execute("SELECT COUNT(*) FROM studies").fetchone()[0]
        trials = connection.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
        if studies < 1 or trials not in {20, 100}:
            raise ValueError(f"Unexpected study/trial count in {path}: {studies}/{trials}")
        return int(trials)
    finally:
        connection.close()


def deserialize_checkpoint(path: Path) -> str:
    if path.suffix in {".pt", ".pth"}:
        import torch

        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        kind = type(value).__name__
        del value
    else:
        with path.open("rb") as handle:
            value = pickle.load(handle)
        kind = type(value).__name__
        del value
    gc.collect()
    return kind


def validate(root: Path, deserialize: bool) -> dict:
    report = {
        "models": 0,
        "databases": 0,
        "trial_counts": {"20": 0, "100": 0},
        "deserialized_models": 0,
        "model_types": {},
    }
    for outcome in OUTCOMES:
        for algorithm in ALGORITHMS:
            for predictors in PREDICTOR_SETS:
                database = root / "optuna" / outcome / algorithm / f"{predictors}.db"
                trials = validate_database(database)
                report["databases"] += 1
                report["trial_counts"][str(trials)] += 1
                for fold in range(5):
                    checkpoint = locate_fold(root, outcome, algorithm, predictors, fold)
                    if checkpoint.stat().st_size == 0:
                        raise ValueError(f"Empty checkpoint: {checkpoint}")
                    report["models"] += 1
                    if deserialize:
                        model_type = deserialize_checkpoint(checkpoint)
                        report["deserialized_models"] += 1
                        report["model_types"][model_type] = report["model_types"].get(model_type, 0) + 1
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--deserialize",
        action="store_true",
        help="Actually load all trusted checkpoints instead of checking structure only.",
    )
    args = parser.parse_args()
    report = validate(args.artifact_dir, args.deserialize)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
