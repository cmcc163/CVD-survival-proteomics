"""Tests for the canonical publication artifact layout."""

from argparse import Namespace
from pathlib import Path

from utils.io_utils import get_output_path
from utils.paths import model_path, optuna_storage, predictor_set, preprocessor_path


def args(tmp_path: Path, data_type: str = "protein") -> Namespace:
    return Namespace(
        artifact_dir=tmp_path,
        disease_type="ASCVD",
        model_name="SAINT",
        data_type=data_type,
        protein_source="lassonet",
    )


def test_predictor_set_names_are_stable(tmp_path) -> None:
    assert predictor_set(args(tmp_path, "classic")) == "classic"
    assert predictor_set(args(tmp_path, "protein")) == "protein_lassonet"
    assert predictor_set(args(tmp_path, "all")) == "all_lassonet"


def test_final_model_and_preprocessor_share_release_directory(tmp_path) -> None:
    configured = args(tmp_path)
    checkpoint = get_output_path(
        configured, "m", "pt", directory="models", extension="best_fold_3"
    )
    transformer = get_output_path(
        configured, "preprocessor", "pkl", directory="models", extension="best_fold_3"
    )

    assert Path(checkpoint) == model_path(configured, 3, ".pt")
    assert Path(transformer) == preprocessor_path(configured, 3)


def test_optuna_database_uses_same_configuration_hierarchy(tmp_path) -> None:
    configured = args(tmp_path)
    uri = optuna_storage(configured, "SAINT_UKB_ASCVD_protein_lassonet")
    expected = (tmp_path / "optuna" / "ASCVD" / "SAINT" / "protein_lassonet.db").resolve()
    assert uri == f"sqlite:///{expected.as_posix()}"
