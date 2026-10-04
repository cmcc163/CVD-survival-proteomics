"""Canonical paths for release models, Optuna studies, and generated runs."""

from __future__ import annotations

import re
from pathlib import Path


_FOLD_EXTENSION = re.compile(r"best_fold_(\d+)$")


def predictor_set(args) -> str:
    """Return the public predictor-set label used throughout the repository."""

    return "classic" if args.data_type == "classic" else f"{args.data_type}_{args.protein_source}"


def model_directory(args) -> Path:
    """Return the one canonical directory for a model configuration."""

    return (
        Path(args.artifact_dir)
        / "models"
        / args.disease_type
        / args.model_name
        / predictor_set(args)
    )


def model_path(args, fold: int, suffix: str) -> Path:
    """Return the canonical path for one released fold checkpoint."""

    return model_directory(args) / f"fold_{fold}{suffix}"


def preprocessor_path(args, fold: int) -> Path:
    """Return the optional demonstration preprocessor path for one fold."""

    return model_directory(args) / f"preprocessor_fold_{fold}.pkl"


def run_directory(args) -> Path:
    """Return the directory for non-release outputs from one configured run."""

    return (
        Path(args.artifact_dir)
        / "runs"
        / args.disease_type
        / args.model_name
        / predictor_set(args)
    )


def canonical_output_path(args, filename, file_type, directory=None, extension=None) -> Path:
    """Resolve legacy save/load calls onto the canonical publication layout.

    The modelling classes still describe final folds as ``best_fold_N``.
    This resolver translates that internal label to the public ``fold_N``
    filename, so training and inference use the same files as the release.
    """

    match = _FOLD_EXTENSION.fullmatch(str(extension)) if extension is not None else None
    if directory == "models":
        base = model_directory(args)
        if match and filename == "m":
            return base / f"fold_{match.group(1)}.{file_type}"
        if match and filename == "preprocessor":
            return base / f"preprocessor_fold_{match.group(1)}.{file_type}"
    else:
        base = run_directory(args)
        if directory:
            base /= directory

    suffix = f"_{extension}" if extension is not None else ""
    return base / f"{filename}{suffix}.{file_type}"


def optuna_storage(args, study_name: str) -> str:
    """Return the SQLite URI for the canonical released Optuna study."""

    name = predictor_set(args)
    if study_name.endswith("_Kneedle"):
        name += "_kneedle"
    database = (
        Path(args.artifact_dir)
        / "optuna"
        / args.disease_type
        / args.model_name
        / f"{name}.db"
    ).resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database.as_posix()}"

