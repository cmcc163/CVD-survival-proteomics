"""Checks for the public aggregate LassoNet feature catalog."""

from pathlib import Path

import pandas as pd


ROOT = Path("features/lassonet")
EXPECTED_SELECTED = {"ASCVD": 71, "HF": 69, "Total_CVD": 83}


def test_outcome_panels_match_authoritative_selection_counts() -> None:
    for outcome, expected in EXPECTED_SELECTED.items():
        panel = pd.read_csv(ROOT / outcome / "lassonet_protein.csv")
        assert len(panel) == expected
        assert panel["Is_Selected"].all()
        assert panel["Selection_Frequency"].ge(0.8).all()


def test_released_model_panels_retain_required_input_dimensions() -> None:
    for outcome, count in EXPECTED_SELECTED.items():
        panel = pd.read_csv(ROOT / outcome / "lassonet_protein.csv")
        assert len(panel) == count
        assert panel["Protein_Name"].is_unique
        assert panel["Is_Selected"].all()


def test_catalog_reports_every_assayed_protein() -> None:
    catalog = pd.read_csv(ROOT / "protein_outcome_catalog.csv")
    assert len(catalog) == 2922
    assert catalog["Protein_Name"].is_unique
    assert set(catalog["Selected_Outcome_Count"].unique()).issubset({0, 1, 2, 3})
