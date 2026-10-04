"""Demonstrate leakage-free preprocessing and Breslow calibration on synthetic data.

The deterministic score below is deliberately not a fitted manuscript model.
It keeps the example lightweight while exercising the same fold-local
preprocessing and absolute-risk calibration interfaces used by the pipeline.
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.baseline_models import process_data
from utils.breslow import BreslowEstimator
from utils.load_data import load_Survial_train_datas


def run(path: Path, folds: int = 5, seed: int = 221) -> dict:
    args = Namespace(path=path, data_type="classic", cat_idx=[], num_features=1)
    x, y, *_rest, cat_idx, feature_names = load_Survial_train_datas(args)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_means = []

    for train_index, test_index in splitter.split(x, y[:, 1]):
        x_train, transformer = process_data(x[train_index], cat_idx, method="deep")
        x_test = process_data(x[test_index], cat_idx, transformer=transformer)

        # A transparent score used only to demonstrate the calibration API.
        train_score = 0.15 * np.asarray(x_train[:, 0], dtype=float)
        test_score = 0.15 * np.asarray(x_test[:, 0], dtype=float)
        breslow = BreslowEstimator().fit(train_score, y[train_index, 1], y[train_index, 0])
        survival = breslow.predict_survival_probabilities(test_score, 3652.0).ravel()
        fold_means.append(float(np.mean(1.0 - survival)))

    return {
        "rows": int(len(x)),
        "features": len(feature_names),
        "folds": folds,
        "mean_demonstration_10y_risk": float(np.mean(fold_means)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("examples/synthetic_survival.csv"))
    args = parser.parse_args()
    result = run(args.path)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
