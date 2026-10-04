"""Tests for the curated model-release importer."""

from __future__ import annotations

import sqlite3

import pytest

from tools.import_release_artifacts import audit_database, audit_model_binary


def make_optuna_like_database(path, value: str = "safe-study") -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE studies (study_id INTEGER, study_name TEXT)")
    connection.execute("CREATE TABLE trials (trial_id INTEGER, state TEXT)")
    connection.execute("INSERT INTO studies VALUES (1, ?)", (value,))
    connection.executemany(
        "INSERT INTO trials VALUES (?, 'COMPLETE')", [(index,) for index in range(100)]
    )
    connection.commit()
    connection.close()


def test_database_audit_reports_trials(tmp_path) -> None:
    database = tmp_path / "study.db"
    make_optuna_like_database(database)

    result = audit_database(database)

    assert result["trials"] == 100
    assert result["trial_states"] == {"COMPLETE": 100}


def test_database_audit_rejects_local_paths(tmp_path) -> None:
    database = tmp_path / "study.db"
    make_optuna_like_database(database, r"C:\Users\researcher\data.csv")

    with pytest.raises(ValueError, match="privacy audit failed"):
        audit_database(database)


def test_model_scan_rejects_embedded_local_paths(tmp_path) -> None:
    model = tmp_path / "model.pkl"
    model.write_bytes(b"header C:\\Users\\researcher\\data\\cohort.csv trailer")

    with pytest.raises(ValueError, match="embedded local path"):
        audit_model_binary(model)
