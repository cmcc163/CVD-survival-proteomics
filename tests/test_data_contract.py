"""Small tests for the public data-loading contract."""

from argparse import Namespace

import pandas as pd
import pytest

from utils.load_data import CLINICAL_FEATURES, load_Survial_train_datas
from models.baseline_models import process_data


def test_clinical_loader_keeps_outcomes_out_of_features(tmp_path):
    rows = []
    for index, ethnic in enumerate([1, 1001, 3, 3001, 2, 4], start=1):
        row = {
            "eid": index,
            "Ethnic": ethnic,
            "time": 1000 + index,
            "Is_Incident": index % 2,
            "total_cholesterol": 5.0,
        }
        row.update({feature: 1.0 for feature in CLINICAL_FEATURES})
        rows.append(row)
    input_file = tmp_path / "endpoint.csv"
    pd.DataFrame(rows).to_csv(input_file, index=False)
    args = Namespace(path=input_file, data_type="classic", num_features=0)

    x_eur, y_eur, x_asian, y_asian, x_other, y_other, cat_idx, names = (
        load_Survial_train_datas(args)
    )

    assert names == [name for name in CLINICAL_FEATURES if name not in {
        "sex", "smoking status", "Diabetes_baseline", "Cholesterol_treatment",
        "hypertension_treatment"
    }] + [
        "sex", "smoking status", "Diabetes_baseline", "Cholesterol_treatment",
        "hypertension_treatment"
    ]
    assert x_eur.shape[0] == y_eur.shape[0] == 2
    assert x_asian.shape[0] == y_asian.shape[0] == 2
    assert x_other.shape[0] == y_other.shape[0] == 2
    assert "eid" not in names and "time" not in names and "Is_Incident" not in names
    assert len(cat_idx) == 5
    assert "smoking status" in names
    pd.read_csv(input_file).drop(columns=["smoking status"]).to_csv(input_file, index=False)
    with pytest.raises(ValueError, match="smoking status"):
        load_Survial_train_datas(args)


def test_deep_encoder_is_fit_on_training_data_only():
    train = pd.DataFrame({"value": [1.0, 2.0], "category": ["A", "B"]}).to_numpy()
    holdout = pd.DataFrame({"value": [3.0], "category": ["UNSEEN"]}).to_numpy()

    _, transformer = process_data(train, [1], method="deep")
    transformed_holdout = process_data(holdout, [1], transformer=transformer)

    # Index zero is reserved for a category not observed in the training data.
    assert transformed_holdout[0, -1] == 0
