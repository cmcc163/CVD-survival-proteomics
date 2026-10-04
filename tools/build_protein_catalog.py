"""Create public LassoNet protein panels and an outcome-overlap catalog.

Only aggregate feature-selection results are copied. Participant-level inputs,
predictions, and identifiers are neither read nor written by this utility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


OUTCOMES = ("ASCVD", "HF", "Total_CVD")
REQUIRED_COLUMNS = (
    "Protein_Name",
    "Selection_Frequency",
    "Selection_Count",
    "Is_Selected",
)


def newest(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {directory}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def read_frequency_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS).difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    result = table.loc[:, REQUIRED_COLUMNS].copy()
    result["Protein_Name"] = result["Protein_Name"].astype(str)
    result["Selection_Frequency"] = pd.to_numeric(result["Selection_Frequency"])
    result["Selection_Count"] = pd.to_numeric(result["Selection_Count"]).astype(int)
    result["Is_Selected"] = result["Is_Selected"].astype(bool)
    if result["Protein_Name"].duplicated().any():
        raise ValueError(f"Duplicate proteins in {path}")
    return result


def build(source_root: Path, destination: Path, model_panel_root: Path | None = None) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    outcome_tables: dict[str, pd.DataFrame] = {}
    provenance: dict[str, dict] = {}

    for outcome in OUTCOMES:
        source_dir = source_root / outcome
        all_path = newest(source_dir, "lassonet_freq_all_proteins_*.csv")
        stats_path = newest(source_dir, "lassonet_freq_stats_*.json")
        table = read_frequency_table(all_path)
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        selected = table.loc[table["Is_Selected"]].copy()
        if len(selected) != int(stats["n_selected"]):
            raise ValueError(f"Selected-protein count disagrees with statistics for {outcome}")

        outcome_dir = destination / outcome
        outcome_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(outcome_dir / "all_protein_frequencies.csv", index=False)
        if model_panel_root is None:
            model_panel = selected.copy()
        else:
            model_names = pd.read_csv(model_panel_root / outcome / "lassonet_protein.csv")
            if "Protein_Name" not in model_names.columns:
                raise ValueError(f"Model input panel for {outcome} has no Protein_Name column")
            model_panel = model_names[["Protein_Name"]].merge(
                table, on="Protein_Name", how="left", validate="one_to_one"
            )
            if model_panel["Selection_Frequency"].isna().any():
                raise ValueError(f"Model input panel contains unknown proteins for {outcome}")
        model_panel.to_csv(outcome_dir / "lassonet_protein.csv", index=False)
        (outcome_dir / "selection_summary.json").write_text(
            json.dumps(stats, indent=2) + "\n", encoding="utf-8"
        )
        outcome_tables[outcome] = table.set_index("Protein_Name")
        provenance[outcome] = {
            "source_type": "repeated-split LassoNet aggregate",
            "n_repeats": int(stats["n_repeats"]),
            "selection_threshold": float(stats["threshold"]),
            "n_selected": int(stats["n_selected"]),
            "model_input_panel_size": int(len(model_panel)),
            "model_panel_currently_selected": int(model_panel["Is_Selected"].sum()),
            "model_panel_matches_selection": bool(
                set(model_panel["Protein_Name"]) == set(selected["Protein_Name"])
            ),
        }

    proteins = sorted(set().union(*(set(table.index) for table in outcome_tables.values())))
    records = []
    for protein in proteins:
        record: dict[str, object] = {"Protein_Name": protein}
        selected_outcomes = []
        for outcome in OUTCOMES:
            table = outcome_tables[outcome]
            if protein in table.index:
                row = table.loc[protein]
                record[f"{outcome}_Frequency"] = float(row["Selection_Frequency"])
                record[f"{outcome}_Count"] = int(row["Selection_Count"])
                selected = bool(row["Is_Selected"])
            else:
                record[f"{outcome}_Frequency"] = 0.0
                record[f"{outcome}_Count"] = 0
                selected = False
            record[f"{outcome}_Selected"] = selected
            if selected:
                selected_outcomes.append(outcome)
        record["Selected_Outcome_Count"] = len(selected_outcomes)
        record["Selected_Outcomes"] = ";".join(selected_outcomes)
        records.append(record)

    pd.DataFrame(records).to_csv(destination / "protein_outcome_catalog.csv", index=False)
    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("features/lassonet"))
    parser.add_argument(
        "--model-panel-root",
        type=Path,
        help="Optional root containing the exact protein panels used by released models.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build(arguments.source_root, arguments.destination, arguments.model_panel_root)
