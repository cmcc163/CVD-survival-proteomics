"""Import only the five best SAINT-MMP12 checkpoints and matching Optuna study."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import optuna
import torch

from scripts.core.saint_mmp12 import FEATURES, OUTCOMES, log_risk, make_model
from tools.import_release_artifacts import audit_database, audit_model_binary, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("artifacts/saint_mmp12"))
    cli = parser.parse_args()
    if cli.destination.exists():
        raise FileExistsError("Destination already exists.")
    records = []
    # Validate every source before copying any files.
    for outcome in OUTCOMES:
        study_name = f"SAINT_UKB_{outcome}_MMP12"
        database = cli.study_root / f"{study_name}.db"
        audit_database(database)
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{database.as_posix()}")
        metadata = dict(artifact="SAINT-MMP12 five-fold survival ensemble", outcome=outcome,
                        feature_names=FEATURES, categorical_indices=list(range(7, 12)),
                        categorical_dimensions=[2] * 5, prediction_batch_size=1024,
                        best_params=study.best_params, study_name=study_name, n_trials=len(study.trials),
                        best_trial_number=study.best_trial.number, best_value=study.best_value,
                        preprocessing_available=False, baseline_hazard_available=False,
                        provenance="Original best-fold checkpoints; old directory labels are not predictor labels",
                        validation={"strict_state_dict_load": True, "synthetic_log_risk_finite": True},
                        files=[])
        files = [(database, Path("optuna_best_params.db"))]
        for fold in range(5):
            weight = cli.weights_root / f"UKB_{outcome}_all_lassonet" / "models" / f"m_best_fold_{fold}.pt"
            audit_model_binary(weight)
            state = torch.load(weight, map_location="cpu", weights_only=True)
            model = make_model(metadata)
            model.model.load_state_dict(state, strict=True)
            scores = log_risk(model, np.zeros((4, len(FEATURES))))
            if not np.isfinite(scores).all():
                raise ValueError(f"Invalid synthetic predictions: {outcome}, fold {fold}")
            files.append((weight, Path("models") / f"fold_{fold}.pt"))
        for source, relative in files:
            metadata["files"].append(dict(path=relative.as_posix(), sha256=sha256(source), bytes=source.stat().st_size))
        records.append((outcome, metadata, files))
    for outcome, metadata, files in records:
        directory = cli.destination / outcome
        for source, relative in files:
            destination = directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Validated and imported 15 best-fold checkpoints and 3 Optuna studies.")


if __name__ == "__main__":
    main()
