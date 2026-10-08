"""Data loading for endpoint-specific UK Biobank survival analyses.

The repository does not distribute participant-level UK Biobank data. This
module expects locally authorised CSV files and never writes participant data.
Feature transformers are deliberately not fitted here. They are fitted inside
each training fold by ``models.baseline_models.process_data`` and then applied
unchanged to validation and external ancestry cohorts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CLINICAL_FEATURES = [
    "age",
    "sex",
    "smoking status",
    "Diabetes_baseline",
    "Cholesterol_treatment",
    "hdl_cholesterol",
    "non_hdl_cholesterol",
    "hypertension_treatment",
    "average_SBP",
    "eGFR",
    "BMI",
]
CATEGORICAL_FEATURES = [
    "sex",
    "smoking status",
    "Diabetes_baseline",
    "Cholesterol_treatment",
    "hypertension_treatment",
]
EUROPEAN_CODES = {1, 1001, 1002, 1003}
ASIAN_CODES = {3, 3001, 3002, 3003, 3004, 5}
RESERVED_COLUMNS = {
    "eid",
    "Ethnic",
    "baseline_date",
    "Event_Date",
    "time",
    "Is_Incident",
    "total_cholesterol",
}
def _read_name_column(path: Path, preferred: Iterable[str]) -> list[str]:
    table = pd.read_csv(path)
    for column in preferred:
        if column in table.columns:
            return table[column].dropna().astype(str).tolist()
    if table.shape[1] == 1:
        return table.iloc[:, 0].dropna().astype(str).tolist()
    raise ValueError(f"No feature-name column found in {path}")


def _protein_panel_path(value: Path) -> Path:
    return value / "lassonet_protein.csv" if value.is_dir() else value


def _prepare_frame(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"eid", "Ethnic", "Is_Incident"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "non_hdl_cholesterol" not in data.columns:
        data["total_cholesterol"] = pd.to_numeric(data["total_cholesterol"], errors="coerce")
        data["hdl_cholesterol"] = pd.to_numeric(data["hdl_cholesterol"], errors="coerce")
        data["non_hdl_cholesterol"] = data["total_cholesterol"] - data["hdl_cholesterol"]

    if "time" not in data.columns:
        baseline = pd.to_datetime(data["baseline_date"], errors="coerce")
        event_date = pd.to_datetime(data["Event_Date"], errors="coerce")
        data["time"] = (event_date - baseline).dt.days

    data["Is_Incident"] = pd.to_numeric(data["Is_Incident"], errors="coerce")
    data["time"] = pd.to_numeric(data["time"], errors="coerce")
    data = data.dropna(subset=["time", "Is_Incident"])
    return data.loc[data["time"] > 0].copy()


def _all_proteins(data: pd.DataFrame, args) -> list[str]:
    if getattr(args, "all_proteins_path", None):
        return _read_name_column(Path(args.all_proteins_path), ["Protein_Name", "protein"])
    excluded = RESERVED_COLUMNS.union(CLINICAL_FEATURES)
    candidates = [column for column in data.columns if column not in excluded]
    if len(candidates) != 2922:
        raise ValueError(
            "Could not infer the 2,922 protein columns safely. Provide --all-proteins-path."
        )
    return candidates


def _selected_proteins(args) -> list[str]:
    value = getattr(args, "protein_path", None)
    if value is None:
        raise ValueError("--protein-path is required for protein and combined predictor sets.")
    return _read_name_column(_protein_panel_path(Path(value)), ["Protein_Name", "protein"])


def _extract_cohorts(data: pd.DataFrame, features: list[str], args):
    missing = [feature for feature in features if feature not in data.columns]
    if missing:
        raise ValueError(f"Input data are missing {len(missing)} requested features: {missing[:10]}")

    categorical = [name for name in CATEGORICAL_FEATURES if name in features]
    continuous = [name for name in features if name not in categorical]
    ordered_features = continuous + categorical
    frame = data[["eid", "Ethnic", "time", "Is_Incident", *ordered_features]].copy()
    for name in categorical:
        frame[name] = frame[name].fillna("Missing").astype(str)

    ethnic = pd.to_numeric(frame["Ethnic"], errors="coerce")
    masks = {
        "european": ethnic.isin(EUROPEAN_CODES),
        "asian": ethnic.isin(ASIAN_CODES),
    }
    masks["other"] = ~(masks["european"] | masks["asian"])

    def arrays(mask):
        subset = frame.loc[mask]
        x = subset[ordered_features].to_numpy()
        y = subset[["time", "Is_Incident"]].to_numpy(dtype=float)
        return x, y

    x_eur, y_eur = arrays(masks["european"])
    x_asian, y_asian = arrays(masks["asian"])
    x_other, y_other = arrays(masks["other"])
    cat_idx = list(range(len(continuous), len(ordered_features)))
    args.num_features = len(ordered_features)
    args.feature_names = ordered_features
    return x_eur, y_eur, x_asian, y_asian, x_other, y_other, cat_idx, ordered_features


def set_category_dimensions(args, x_training: np.ndarray) -> None:
    """Infer training-only cardinalities and reserve index zero for unknowns."""

    args.cat_dims = [len(pd.unique(x_training[:, index])) + 1 for index in args.cat_idx]


def load_datas(args, data_type="classic"):
    """Load all proteins for stability LassoNet selection."""

    data = _prepare_frame(Path(args.path))
    proteins = _all_proteins(data, args)
    if data_type == "all":
        features = proteins + CLINICAL_FEATURES
    elif data_type == "protein":
        features = proteins
    else:
        features = CLINICAL_FEATURES
    return _extract_cohorts(data, features, args)


def load_Survial_train_datas(args, protein_source="lassonet"):
    """Load one manuscript predictor configuration using LassoNet panels only."""

    if protein_source not in {"lassonet", "NA", None}:
        raise ValueError("Only LassoNet-selected protein panels are supported.")
    data = _prepare_frame(Path(args.path))
    if args.data_type == "classic":
        features = CLINICAL_FEATURES
    elif args.data_type == "protein":
        features = _selected_proteins(args)
    else:
        features = _selected_proteins(args) + CLINICAL_FEATURES
    return _extract_cohorts(data, features, args)


def load_Survial_train_datas_kneedle(args, protein_source="lassonet"):
    """Load the Kneedle-selected reduced predictor panel."""

    if not getattr(args, "reduced_panel_path", None):
        raise ValueError("--reduced-panel-path is required for RP-SAINT.")
    data = _prepare_frame(Path(args.path))
    features = _read_name_column(
        Path(args.reduced_panel_path),
        ["Feature", "feature", "Feature_Name", "variable"],
    )
    return _extract_cohorts(data, features, args)
