"""End-to-end checks using only deterministic synthetic observations."""

from argparse import Namespace

import numpy as np

from examples.generate_synthetic_data import generate
from models.baseline_models import process_data
from utils.breslow import BreslowEstimator
from utils.load_data import load_Survial_train_datas, set_category_dimensions


def test_synthetic_load_preprocess_and_calibrate(tmp_path) -> None:
    data_path = tmp_path / "synthetic.csv"
    generate(240, 221).to_csv(data_path, index=False)
    args = Namespace(path=data_path, data_type="classic", cat_idx=[], num_features=1)

    x_eur, y_eur, x_asian, y_asian, x_other, y_other, cat_idx, names = (
        load_Survial_train_datas(args)
    )
    assert len(names) == 11
    assert len(x_eur) and len(x_asian) and len(x_other)
    args.cat_idx = cat_idx
    set_category_dimensions(args, x_eur)

    transformed, transformer = process_data(x_eur, cat_idx, method="deep")
    transformed_asian = process_data(x_asian, cat_idx, transformer=transformer)
    assert transformed.shape[1] == transformed_asian.shape[1] == 11
    assert np.isfinite(transformed.astype(float)).all()

    log_risk = 0.03 * transformed[:, 0].astype(float)
    estimator = BreslowEstimator().fit(log_risk, y_eur[:, 1], y_eur[:, 0])
    probabilities = estimator.predict_survival_probabilities(log_risk[:5], [3652.0])
    assert probabilities.shape == (5, 1)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    functions = estimator.get_survival_function(log_risk[:2])
    assert len(functions) == 2
    assert 0 <= functions[0](3652.0) <= 1
