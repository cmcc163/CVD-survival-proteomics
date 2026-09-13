from __future__ import annotations

import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index


PRED_DIR = Path(os.environ.get("CVD_REDUCED_PREDICTION_ROOT", "artifacts/predictions/Kneedle/SAINT"))
OUTPUT_MODEL_DIR = Path(os.environ.get("CVD_REDUCED_MODEL_OUTPUT", "artifacts/models/SAINT"))
OUT_DIR = Path(__file__).resolve().parent

OUTCOMES = ["Total_CVD", "ASCVD", "HF"]
OUTCOME_LABELS = {"Total_CVD": "Total CVD", "ASCVD": "ASCVD", "HF": "HF"}
TARGET_TIME = 365.25 * 10


def parse_ci(results_txt: Path, metric_regex: str) -> tuple[float, float]:
    text = results_txt.read_text(encoding="utf-8", errors="ignore")
    pattern = rf"european_{metric_regex}.*?95% CI:\s*\(([^,]+),\s*([^)]+)\)"
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not parse {metric_regex} CI from {results_txt}")
    return float(match.group(1)), float(match.group(2))


def format_metric(point: float, ci: tuple[float, float]) -> str:
    return f"{point:.4f} ({ci[0]:.4f}-{ci[1]:.4f})"


def censoring_survival_at(kmf: KaplanMeierFitter, times: np.ndarray) -> np.ndarray:
    survival = kmf.survival_function_[kmf.survival_function_.columns[0]]
    return np.asarray(survival.reindex(times, method="ffill").fillna(1.0), dtype=float)


def c_index_from_predictions(path: Path) -> float:
    data = pd.read_csv(path, usecols=["time", "event", "risk_score"])
    data = data.dropna(subset=["time", "event", "risk_score"])
    return float(concordance_index(data["time"], -data["risk_score"], data["event"]))


def cumulative_dynamic_auc_10yr(path: Path) -> float:
    data = pd.read_csv(path, usecols=["time", "event", "prob_10yr"])
    data = data.dropna(subset=["time", "event", "prob_10yr"]).copy()
    data["time"] = pd.to_numeric(data["time"], errors="coerce")
    data["event"] = pd.to_numeric(data["event"], errors="coerce")
    data["prob_10yr"] = pd.to_numeric(data["prob_10yr"], errors="coerce")
    data = data.dropna()

    cases = data[(data["event"].eq(1)) & (data["time"] <= TARGET_TIME)].copy()
    controls = data[data["time"] > TARGET_TIME].copy()
    if cases.empty or controls.empty:
        return math.nan

    kmf = KaplanMeierFitter()
    kmf.fit(data["time"], event_observed=1 - data["event"])
    case_weights = 1.0 / np.clip(censoring_survival_at(kmf, cases["time"].to_numpy()), 1e-8, None)
    case_scores = cases["prob_10yr"].to_numpy()
    control_scores = controls["prob_10yr"].to_numpy()

    total = 0.0
    for score, weight in zip(case_scores, case_weights):
        total += weight * (
            np.sum(score > control_scores) + 0.5 * np.sum(np.isclose(score, control_scores))
        )
    return float(total / (np.sum(case_weights) * len(control_scores)))


def results_file(outcome: str) -> Path:
    return OUTPUT_MODEL_DIR / f"UKB_{outcome}_all_lassonet" / "results.txt"


def main() -> None:
    rows = []
    for outcome in OUTCOMES:
        pred_file = PRED_DIR / outcome / "all_lassonet_val.csv"
        result_txt = results_file(outcome)
        c_index = c_index_from_predictions(pred_file)
        auc10 = cumulative_dynamic_auc_10yr(pred_file)
        c_ci = parse_ci(result_txt, "C-index")
        auc_ci = parse_ci(result_txt, "AUC \\(10yr\\)")
        rows.append(
            {
                "Outcome": OUTCOME_LABELS[outcome],
                "Outcome key": outcome,
                "Validation set": "European hold-out set",
                "Model": "SAINT reduced predictor panel",
                "Predictor set": "Kneedle-selected clinical plus protein markers",
                "C-index": c_index,
                "C-index lower 95% CI": c_ci[0],
                "C-index upper 95% CI": c_ci[1],
                "C-index formatted": format_metric(c_index, c_ci),
                "AUC 10yr": auc10,
                "AUC 10yr lower 95% CI": auc_ci[0],
                "AUC 10yr upper 95% CI": auc_ci[1],
                "AUC 10yr formatted": format_metric(auc10, auc_ci),
                "Prediction file": str(pred_file),
                "CI source": str(result_txt),
            }
        )

    out_xlsx = OUT_DIR / "reduced_saint_discrimination_python.xlsx"
    out_csv = OUT_DIR / "reduced_saint_discrimination_python.csv"
    pd.DataFrame(rows).to_excel(out_xlsx, index=False)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved {out_xlsx}")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
