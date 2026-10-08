"""Build a privacy-audited release bundle from completed training runs.

The importer does not deserialize model files. It copies only the best
fold-specific checkpoints and the matching Optuna databases, computes SHA-256
checksums, and records missing expected artifacts in a public manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


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
MODEL_SUFFIXES = (".pt", ".pth", ".pkl", ".joblib")
SENSITIVE_COLUMN = re.compile(
    r"(?:^|_)(?:eid|participant|patient|subject|person|nhs|email|phone|address)(?:_|$)",
    re.IGNORECASE,
)
LOCAL_PATH = re.compile(
    r"(?:[A-Za-z]:\\|/home/|/Users/|\\\\|\.csv\b|\.xlsx\b)",
    re.IGNORECASE,
)
LOCAL_PATH_BYTES = re.compile(
    rb"(?:[A-Za-z]:\\(?:Users|Desktop|data)\\|/home/[^/\x00]+/|/Users/[^/\x00]+/)",
    re.IGNORECASE,
)
ACCEPTED_OPTUNA_TRIALS = (20, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root containing ASCVD, HF, and Total_CVD result directories.",
    )
    parser.add_argument(
        "--legacy-output-root",
        type=Path,
        help="Optional legacy output directory used only to fill missing models.",
    )
    parser.add_argument("--destination", type=Path, default=Path("artifacts"))
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_database(path: Path) -> dict:
    """Audit Optuna SQLite text and schema without modifying the database."""

    findings: list[str] = []
    row_count = 0
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            quoted_table = table.replace('"', '""')
            columns = list(connection.execute(f'PRAGMA table_info("{quoted_table}")'))
            for column in columns:
                name = str(column[1])
                if SENSITIVE_COLUMN.search(name):
                    findings.append(f"sensitive column name: {table}.{name}")
            row_count += int(
                connection.execute(f'SELECT COUNT(*) FROM "{quoted_table}"').fetchone()[0]
            )
            for column in columns:
                name, declared_type = str(column[1]), str(column[2]).upper()
                if declared_type not in {"", "TEXT"}:
                    continue
                quoted_name = name.replace('"', '""')
                query = (
                    f'SELECT "{quoted_name}" FROM "{quoted_table}" '
                    f'WHERE "{quoted_name}" IS NOT NULL'
                )
                for (value,) in connection.execute(query):
                    if isinstance(value, str) and LOCAL_PATH.search(value):
                        findings.append(f"local path-like value: {table}.{name}")
                        break
    finally:
        connection.close()
    if findings:
        raise ValueError(f"database privacy audit failed for {path.name}: {findings}")
    trial_count = 0
    trial_states: dict[str, int] = {}
    if "trials" in tables:
        trial_count = int(connection_trial_count(path))
        trial_states = database_trial_states(path)
    return {
        "tables": len(tables),
        "rows": row_count,
        "trials": trial_count,
        "trial_states": trial_states,
    }


def connection_trial_count(path: Path) -> int:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return int(connection.execute("SELECT COUNT(*) FROM trials").fetchone()[0])
    finally:
        connection.close()


def database_trial_states(path: Path) -> dict[str, int]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return {
            str(state): int(count)
            for state, count in connection.execute(
                "SELECT state, COUNT(*) FROM trials GROUP BY state"
            )
        }
    finally:
        connection.close()


def audit_model_binary(path: Path) -> None:
    """Reject obvious embedded workstation paths without deserializing a model."""

    overlap = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sample = overlap + block
            if LOCAL_PATH_BYTES.search(sample):
                raise ValueError(f"embedded local path found in model: {path.name}")
            overlap = sample[-256:]


def result_run_dir(root: Path, outcome: str, algorithm: str, predictor_set: str) -> Path:
    return root / outcome / algorithm / f"UKB_{outcome}_{predictor_set}"


def legacy_run_dir(root: Path, outcome: str, algorithm: str, predictor_set: str) -> Path:
    return root / algorithm / f"UKB_{outcome}_{predictor_set}"


def find_model(
    roots: list[Path], outcome: str, algorithm: str, predictor_set: str, fold: int
) -> Path | None:
    for run_dir in roots:
        model_dir = run_dir / "models"
        for suffix in MODEL_SUFFIXES:
            candidate = model_dir / f"m_best_fold_{fold}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def find_database(root: Path, outcome: str, algorithm: str, predictor_set: str) -> Path | None:
    algorithm_dir = root / outcome / algorithm
    exact = algorithm_dir / f"{algorithm}_UKB_{outcome}_{predictor_set}.db"
    if exact.is_file():
        return exact
    matches = sorted(algorithm_dir.glob(f"*_UKB_{outcome}_{predictor_set}.db"))
    return matches[0] if len(matches) == 1 else None


def copy_with_record(
    source: Path, destination: Path, release_root: Path, kind: str, metadata: dict
) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return record_file(destination, release_root, kind, metadata)


def record_file(path: Path, release_root: Path, kind: str, metadata: dict) -> dict:
    """Record a release file without exposing its original workstation path."""

    return {
        "kind": kind,
        "path": path.relative_to(release_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        **metadata,
    }


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    destination = args.destination.resolve()
    legacy_root = args.legacy_output_root.resolve() if args.legacy_output_root else None
    if not results_root.is_dir():
        raise FileNotFoundError(results_root)

    files: list[dict] = []
    missing: list[dict] = []
    db_audit = {"files": 0, "tables": 0, "rows": 0}
    trial_mismatches: list[dict] = []
    audited_models = 0

    for outcome in OUTCOMES:
        for algorithm in ALGORITHMS:
            for predictor_set in PREDICTOR_SETS:
                db_source = find_database(results_root, outcome, algorithm, predictor_set)
                if db_source is None:
                    missing.append(
                        {
                            "kind": "optuna_database",
                            "outcome": outcome,
                            "algorithm": algorithm,
                            "predictor_set": predictor_set,
                        }
                    )
                else:
                    audit = audit_database(db_source)
                    db_audit["files"] += 1
                    db_audit["tables"] += audit["tables"]
                    db_audit["rows"] += audit["rows"]
                    if audit["trials"] not in ACCEPTED_OPTUNA_TRIALS:
                        trial_mismatches.append(
                            {
                                "outcome": outcome,
                                "algorithm": algorithm,
                                "predictor_set": predictor_set,
                                "accepted_trials": list(ACCEPTED_OPTUNA_TRIALS),
                                "observed_trials": audit["trials"],
                            }
                        )
                    target = destination / "optuna" / outcome / algorithm / f"{predictor_set}.db"
                    files.append(
                        copy_with_record(
                            db_source,
                            target,
                            destination,
                            "optuna_database",
                            {
                                "outcome": outcome,
                                "algorithm": algorithm,
                                "predictor_set": predictor_set,
                                "trial_count": audit["trials"],
                                "trial_states": audit["trial_states"],
                            },
                        )
                    )

                run_roots = [result_run_dir(results_root, outcome, algorithm, predictor_set)]
                if legacy_root is not None:
                    run_roots.append(legacy_run_dir(legacy_root, outcome, algorithm, predictor_set))
                for fold in range(5):
                    model_source = find_model(run_roots, outcome, algorithm, predictor_set, fold)
                    release_dir = destination / "models" / outcome / algorithm / predictor_set
                    existing = next(
                        (
                            release_dir / f"fold_{fold}{suffix}"
                            for suffix in MODEL_SUFFIXES
                            if (release_dir / f"fold_{fold}{suffix}").is_file()
                        ),
                        None,
                    )
                    if model_source is None:
                        if existing is not None:
                            audit_model_binary(existing)
                            audited_models += 1
                            files.append(
                                record_file(
                                    existing,
                                    destination,
                                    "fold_model",
                                    {
                                        "outcome": outcome,
                                        "algorithm": algorithm,
                                        "predictor_set": predictor_set,
                                        "fold": fold,
                                    },
                                )
                            )
                            continue
                        missing.append(
                            {
                                "kind": "fold_model",
                                "outcome": outcome,
                                "algorithm": algorithm,
                                "predictor_set": predictor_set,
                                "fold": fold,
                            }
                        )
                        continue
                    target = release_dir / f"fold_{fold}{model_source.suffix.lower()}"
                    audit_model_binary(model_source)
                    audited_models += 1
                    files.append(
                        copy_with_record(
                            model_source,
                            target,
                            destination,
                            "fold_model",
                            {
                                "outcome": outcome,
                                "algorithm": algorithm,
                                "predictor_set": predictor_set,
                                "fold": fold,
                            },
                        )
                    )

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "release_scope": {
            "outcomes": list(OUTCOMES),
            "algorithms": list(ALGORITHMS),
            "predictor_sets": list(PREDICTOR_SETS),
            "folds": list(range(5)),
        },
        "privacy_audit": {
            "optuna_sqlite": db_audit,
            "model_binary_path_scan": {"files": audited_models},
            "status": "passed",
            "model_deserialization_performed": False,
        },
        "summary": {
            "released_files": len(files),
            "released_bytes": sum(item["bytes"] for item in files),
            "missing_expected_files": len(missing),
            "complete": not missing,
            "release_ready": not missing and not trial_mismatches,
        },
        "files": sorted(files, key=lambda item: item["path"]),
        "missing": missing,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest["summary"], indent=2))
    print(f"Manifest: {manifest_path}")
    if missing and not args.allow_incomplete:
        raise SystemExit(
            "Release is incomplete. Inspect manifest.json or rerun with --allow-incomplete."
        )


if __name__ == "__main__":
    main()
