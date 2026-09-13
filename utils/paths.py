"""Portable paths for generated artifacts."""

from pathlib import Path


def optuna_storage(args, study_name: str) -> str:
    """Return a SQLite URI rooted in the configured artifact directory."""

    directory = Path(args.artifact_dir) / "optuna"
    directory.mkdir(parents=True, exist_ok=True)
    database = (directory / f"{study_name}.db").resolve()
    return f"sqlite:///{database.as_posix()}"

