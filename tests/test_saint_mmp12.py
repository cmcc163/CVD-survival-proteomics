"""Check the MMP12 scorer against the shared SAINT survival implementation."""
import numpy as np
import pytest

from scripts.core.saint_mmp12 import FEATURES, log_risk, make_model


def test_scorer_matches_shared_model():
    model = make_model(dict(outcome="Total_CVD", categorical_dimensions=[2] * 5,
                            best_params=dict(dim=8, depth=1, heads=8, dropout=0.5)))
    rng = np.random.default_rng(221)
    x = rng.normal(size=(9, len(FEATURES)))
    x[:, 7:] = rng.integers(0, 2, size=(9, 5))
    expected, _ = model.predict(x)
    np.testing.assert_allclose(log_risk(model, x), expected, rtol=1e-6, atol=1e-7)
    x[0, 7] = 2
    with pytest.raises(ValueError, match="Categorical codes"):
        log_risk(model, x)


def test_refit_and_raw_prediction(tmp_path):
    import json
    import pandas as pd
    from scripts.core.saint_mmp12 import main
    rng = np.random.default_rng(221)
    frame = pd.DataFrame(rng.normal(size=(60, 12)), columns=FEATURES)
    for name in FEATURES[7:]:
        frame[name] = rng.integers(0, 2, size=len(frame))
    frame["eid"] = np.arange(60)
    frame["Ethnic"] = 1
    frame["time"] = rng.uniform(100, 4000, size=60)
    frame["Is_Incident"] = np.tile([0, 1], 30)
    input_file = tmp_path / "synthetic.csv"
    frame.to_csv(input_file, index=False)
    seed_root = tmp_path / "seed"
    (seed_root / "Total_CVD").mkdir(parents=True)
    metadata = dict(outcome="Total_CVD", categorical_dimensions=[2] * 5,
                    prediction_batch_size=1024, best_params=dict(dim=8, depth=1, heads=8, dropout=0.5))
    (seed_root / "Total_CVD" / "metadata.json").write_text(json.dumps(metadata))
    trained = tmp_path / "trained"
    main(["train", "--outcome", "Total_CVD", "--input", str(input_file),
          "--output", str(trained), "--artifact-root", str(seed_root), "--epochs", "1"])
    output = tmp_path / "prediction.csv"
    main(["predict", "--outcome", "Total_CVD", "--input", str(input_file),
          "--output", str(output), "--artifact-root", str(trained)])
    assert np.isfinite(pd.read_csv(output).log_risk).all()
    assert len(list((trained / "Total_CVD" / "models").glob("fold_*.pt"))) == 5
