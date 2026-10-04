"""Generate a deterministic, non-participant demonstration survival dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.load_data import CLINICAL_FEATURES


def generate(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(40, 75, rows)
    sex = rng.integers(0, 2, rows)
    sbp = rng.normal(130, 15, rows)
    linear_risk = 0.04 * (age - 55) + 0.25 * sex + 0.012 * (sbp - 125)
    event_time = rng.exponential(5000 / np.exp(linear_risk))
    censor_time = rng.uniform(2500, 7500, rows)
    observed_time = np.maximum(1, np.minimum(event_time, censor_time)).round().astype(int)

    frame = pd.DataFrame(
        {
            "eid": [f"SYNTHETIC_{index:04d}" for index in range(rows)],
            "Ethnic": rng.choice([1, 1001, 3, 3001, 4], rows, p=[0.55, 0.15, 0.1, 0.1, 0.1]),
            "time": observed_time,
            "Is_Incident": (event_time <= censor_time).astype(int),
            "age": age,
            "sex": sex,
            "ever_smoked": rng.integers(0, 2, rows),
            "Diabetes_baseline": rng.binomial(1, 0.12, rows),
            "Cholesterol_treatment": rng.binomial(1, 0.25, rows),
            "hdl_cholesterol": rng.normal(1.4, 0.3, rows).clip(0.4, None),
            "non_hdl_cholesterol": rng.normal(3.5, 0.8, rows).clip(0.5, None),
            "hypertension_treatment": rng.binomial(1, 0.3, rows),
            "average_SBP": sbp,
            "eGFR": rng.normal(88, 14, rows).clip(20, 140),
            "BMI": rng.normal(27, 4, rows).clip(16, 50),
        }
    )
    return frame[["eid", "Ethnic", "time", "Is_Incident", *CLINICAL_FEATURES]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/synthetic_survival.csv"))
    parser.add_argument("--rows", type=int, default=240)
    parser.add_argument("--seed", type=int, default=221)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate(args.rows, args.seed).to_csv(args.output, index=False)
    print(f"Wrote {args.rows} synthetic rows to {args.output}")


if __name__ == "__main__":
    main()
