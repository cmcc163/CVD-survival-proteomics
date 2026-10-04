"""Recover the reduced-panel (RP) SAINT survival ensembles.

This utility deliberately reads the frozen Optuna studies and never launches new
trials.  It reproduces the original split/training/prediction path, saves the five
fold weights in an isolated release directory, and compares regenerated
predictions with the historical RP-SAINT predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split


OUTCOMES = ("Total_CVD", "ASCVD", "HF")
SPLITS = ("train", "val", "asian", "other")
PREDICTION_COLUMNS = ("risk_score", "prob_10yr", "prob_15yr", "prob_20yr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--outcome",
        action="append",
        choices=OUTCOMES,
        help="Outcome to train; repeat for multiple outcomes (default: all).",
    )
    parser.add_argument(
        "--keep-regenerated-predictions",
        action="store_true",
        help="Keep participant-level regenerated predictions locally.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Rebuild validation_summary.json from completed metadata files.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def compare_prediction_frames(reference: Path, regenerated: pd.DataFrame) -> dict[str, Any]:
    historical = pd.read_csv(reference, usecols=["time", "event", *PREDICTION_COLUMNS])
    result: dict[str, Any] = {
        "reference_file": reference.name,
        "reference_sha256": sha256(reference),
        "n_reference": len(historical),
        "n_regenerated": len(regenerated),
        "row_count_match": len(historical) == len(regenerated),
    }
    if len(historical) != len(regenerated):
        return result

    result["outcome_time_max_abs_diff"] = float(
        np.max(np.abs(historical["time"].to_numpy() - regenerated["time"].to_numpy()))
    )
    result["outcome_event_exact_match"] = bool(
        np.array_equal(historical["event"].to_numpy(), regenerated["event"].to_numpy())
    )
    for column in PREDICTION_COLUMNS:
        left = historical[column].to_numpy(dtype=float)
        right = regenerated[column].to_numpy(dtype=float)
        delta = np.abs(left - right)
        result[column] = {
            "max_abs_diff": float(np.max(delta)),
            "mean_abs_diff": float(np.mean(delta)),
            "pearson_r": float(np.corrcoef(left, right)[0, 1]),
            "allclose_rtol_1e-5_atol_1e-7": bool(
                np.allclose(left, right, rtol=1e-5, atol=1e-7)
            ),
        }
    return result


def build_result_frame(y: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": y[:, 0],
            "event": y[:, 1].astype(int),
            **predictions,
        }
    )


def train_outcome(cli: argparse.Namespace, outcome: str) -> dict[str, Any]:
    source_root = cli.source_root.resolve()
    output_dir = (cli.output_root / outcome).resolve()
    work_dir = output_dir / "_work"
    model_dir = output_dir / "models"
    baseline_dir = output_dir / "breslow"
    report_dir = output_dir / "reports"
    for directory in (work_dir, model_dir, baseline_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # The original loader resolves the Kneedle workbook relative to the project.
    os.chdir(source_root)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    from models import str2model
    from models.baseline_models import process_data
    import models.basemodel_torch as basemodel_torch
    from utils.load_data import load_Survial_train_datas_kneedle
    from utils.parser import get_parser

    def isolated_output_path(
        model_args: argparse.Namespace,
        filename: str,
        file_type: str,
        directory: str | None = None,
        extension: str | None = None,
    ) -> str:
        target_dir = work_dir / (directory or "")
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{extension}" if extension is not None else ""
        return str(target_dir / f"{filename}{suffix}.{file_type}")

    # SAINT.fit_survival calls the module-level helper imported by BaseModelTorch.
    # Redirecting only this binding isolates temporary checkpoints from old runs.
    basemodel_torch.get_output_path = isolated_output_path

    data_path = cli.data_root / outcome / "followup_incident_20260122.csv"
    protein_dir = cli.data_root / outcome
    db_path = source_root / f"SAINT_UKB_{outcome}_all_lassonet_Kneedle.db"
    historical_dir = source_root / "saved_predictions" / "Kneedle" / "SAINT" / outcome
    required = [data_path, protein_dir / "lassonet_protein.csv", db_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs for {outcome}: {missing}")

    project_parser = get_parser()
    model_args = project_parser.parse_args(
        args=[
            "--config",
            str(source_root / "config" / "UKB.yml"),
            "--path",
            str(data_path),
            "--protein_path",
            str(protein_dir) + os.sep,
        ]
    )
    model_args.disease_type = outcome
    model_args.data_type = "all"
    model_args.protein_source = "lassonet"
    model_args.study_name = f"SAINT_UKB_{outcome}_all_lassonet_Kneedle"

    storage = f"sqlite:///{db_path.as_posix()}"
    study = optuna.load_study(study_name=model_args.study_name, storage=storage)
    best_trial = study.best_trial
    model_args.best_n_epochs = best_trial.user_attrs.get("best_n_epochs")

    (
        x_eur,
        y_eur,
        x_asian,
        y_asian,
        x_other,
        y_other,
        cat_idx,
        feature_names,
    ) = load_Survial_train_datas_kneedle(model_args, protein_source="lassonet")
    model_args.feature_names = feature_names
    model_args.cat_idx = cat_idx

    x_train, x_val, y_train, y_val = train_test_split(
        x_eur,
        y_eur,
        test_size=model_args.ratio,
        random_state=model_args.seed,
        stratify=y_eur[:, 1],
    )

    model_class = str2model(model_args.model_name)
    base_model = model_class(best_trial.params, model_args)
    splitter = StratifiedKFold(
        n_splits=model_args.num_splits,
        shuffle=model_args.shuffle,
        random_state=model_args.seed,
    )

    fold_records: list[dict[str, Any]] = []
    for fold, (fit_index, stop_index) in enumerate(
        splitter.split(x_train, y_train[:, 1].astype(int))
    ):
        print(f"\n[{outcome}] training fold {fold + 1}/{model_args.num_splits}", flush=True)
        x_fit, transformer = process_data(x_train[fit_index], cat_idx, method="deep")
        x_stop = process_data(x_train[stop_index], cat_idx, transformer=transformer)
        fold_model = base_model.clone()
        _, validation_history, best_epoch = fold_model.fit(
            x_fit,
            y_train[fit_index],
            x_stop,
            y_train[stop_index],
            fold,
        )
        weight_path = model_dir / f"fold_{fold}.pt"
        torch.save(fold_model.model.state_dict(), weight_path)
        fold_records.append(
            {
                "fold": fold,
                "n_fit": len(fit_index),
                "n_early_stopping": len(stop_index),
                "best_epoch_zero_based": int(best_epoch),
                "best_validation_c_index": float(np.max(validation_history)),
                "weight_file": str(weight_path.relative_to(cli.output_root)),
                "weight_sha256": sha256(weight_path),
            }
        )
        del fold_model, x_fit, x_stop, transformer
        torch.cuda.empty_cache()

    # Preserve the exact full-training preprocessing used by the historical
    # ensemble prediction script.
    x_train_p, inference_transformer = process_data(x_train, cat_idx, method="deep")
    transformed = {
        "train": x_train_p,
        "val": process_data(x_val, cat_idx, transformer=inference_transformer),
        "asian": process_data(x_asian, cat_idx, transformer=inference_transformer),
        "other": process_data(x_other, cat_idx, transformer=inference_transformer),
    }
    outcomes = {"train": y_train, "val": y_val, "asian": y_asian, "other": y_other}
    preprocessor_path = output_dir / "preprocessor.pkl"
    with preprocessor_path.open("wb") as handle:
        pickle.dump(inference_transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    accumulators = {
        split: {column: np.zeros(len(values), dtype=np.float64) for column in PREDICTION_COLUMNS}
        for split, values in outcomes.items()
    }
    for fold in range(model_args.num_splits):
        print(f"[{outcome}] validating fold {fold + 1}/{model_args.num_splits}", flush=True)
        fold_model = model_class(best_trial.params, model_args)
        state = torch.load(model_dir / f"fold_{fold}.pt", map_location=fold_model.device, weights_only=True)
        fold_model.model.load_state_dict(state)
        fold_model.model.to(fold_model.device)
        fold_model.fit_breslow(x_train_p, y_train)
        baseline_path = baseline_dir / f"fold_{fold}.pkl"
        with baseline_path.open("wb") as handle:
            pickle.dump(fold_model.breslow, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for split, x_values in transformed.items():
            risk, prob10 = fold_model.predict(x_values)
            accumulators[split]["risk_score"] += risk
            accumulators[split]["prob_10yr"] += prob10
            for years in (15, 20):
                probability = 1.0 - fold_model.breslow.predict_survival_probabilities(
                    risk, years * 365.2
                ).flatten()
                accumulators[split][f"prob_{years}yr"] += probability
        del fold_model, state
        torch.cuda.empty_cache()

    comparison: dict[str, Any] = {}
    prediction_dir = output_dir / "regenerated_predictions"
    if cli.keep_regenerated_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        averaged = {
            column: values / model_args.num_splits
            for column, values in accumulators[split].items()
        }
        regenerated = build_result_frame(outcomes[split], averaged)
        reference = historical_dir / f"all_lassonet_{split}.csv"
        comparison[split] = compare_prediction_frames(reference, regenerated)
        if cli.keep_regenerated_predictions:
            regenerated.to_csv(prediction_dir / f"{split}.csv", index=False)

    copied_db = output_dir / "optuna_best_params.db"
    shutil.copy2(db_path, copied_db)
    metadata = {
        "artifact": "RP-SAINT five-fold survival ensemble",
        "outcome": outcome,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_code": {
            "saint_py_sha256": sha256(source_root / "models" / "saint.py"),
            "load_data_py_sha256": sha256(source_root / "utils" / "load_data.py"),
            "config_sha256": sha256(source_root / "config" / "UKB.yml"),
        },
        "optuna": {
            "study_name": model_args.study_name,
            "database_sha256": sha256(db_path),
            "n_trials": len(study.trials),
            "best_trial_number": best_trial.number,
            "best_value": best_trial.value,
            "best_params": best_trial.params,
            "best_n_epochs_user_attr": model_args.best_n_epochs,
        },
        "training": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "seed": model_args.seed,
            "holdout_ratio": model_args.ratio,
            "folds": model_args.num_splits,
            "batch_size": model_args.batch_size,
            "early_stopping_rounds": model_args.early_stopping_rounds,
            "feature_names": feature_names,
            "categorical_indices": cat_idx,
            "categorical_dimensions": model_args.cat_dims,
            "sample_sizes": {split: len(values) for split, values in outcomes.items()},
            "fold_details": fold_records,
        },
        "release_files": {
            "preprocessor_sha256": sha256(preprocessor_path),
            "copied_optuna_database_sha256": sha256(copied_db),
        },
        "historical_prediction_comparison": comparison,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(json_ready(metadata), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"[{outcome}] completed: {metadata_path}", flush=True)
    return metadata


def main() -> None:
    cli = parse_args()
    cli.source_root = cli.source_root.resolve()
    cli.data_root = cli.data_root.resolve()
    cli.output_root = cli.output_root.resolve()
    cli.output_root.mkdir(parents=True, exist_ok=True)
    outcomes = cli.outcome or list(OUTCOMES)
    if not cli.summary_only:
        for outcome in outcomes:
            train_outcome(cli, outcome)
    # Always summarize every completed outcome, including outcomes produced by
    # an earlier invocation of this resumable utility.
    summaries = []
    for outcome in OUTCOMES:
        metadata_path = cli.output_root / outcome / "metadata.json"
        if metadata_path.exists():
            summaries.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    summary_path = cli.output_root / "validation_summary.json"
    summary_path.write_text(
        json.dumps(json_ready(summaries), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"All requested outcomes completed: {summary_path}")


if __name__ == "__main__":
    main()
