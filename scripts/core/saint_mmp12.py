"""Train clinical-plus-MMP12 SAINT or score a released five-fold ensemble."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.baseline_models import process_data
from models.saint import SAINT
from utils.load_data import EUROPEAN_CODES, _prepare_frame
from utils.parser import get_parser

# Preserve the feature order used by the original MMP12 training loader.
FEATURES = ["age", "hdl_cholesterol", "non_hdl_cholesterol", "average_SBP",
            "eGFR", "BMI", "MMP12", "sex", "smoking status", "Diabetes_baseline",
            "hypertension_treatment", "Cholesterol_treatment"]
CAT_INDICES = list(range(7, 12))
OUTCOMES = ("Total_CVD", "ASCVD", "HF")


def make_model(metadata, use_gpu=False, batch_size=1024, cat_dims=None):
    """Instantiate the shared SAINT survival architecture, without loading data."""
    args = get_parser().parse_args([
        "--disease-type", metadata["outcome"], "--data-type", "all",
        "--model-name", "SAINT", "--path", "unused.csv", "--no-use-gpu"])
    args.use_gpu = use_gpu
    args.num_features = len(FEATURES)
    args.num_classes = 1
    args.cat_idx = CAT_INDICES
    args.cat_dims = cat_dims or metadata["categorical_dimensions"]
    args.val_batch_size = args.batch_size = batch_size
    return SAINT(metadata["best_params"], args)


def log_risk(model, values):
    """Return log hazards only; never substitute zeros for missing calibration."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(FEATURES) or not len(values):
        raise ValueError("Expected a nonempty matrix with twelve predictors.")
    if not np.isfinite(values).all():
        raise ValueError("Predictors must be finite; impute missing values before scoring.")
    cats = values[:, CAT_INDICES]
    if np.any(cats != np.floor(cats)) or np.any(cats < 0) or np.any(cats >= model.args.cat_dims):
        raise ValueError("Categorical codes do not match the model embedding dimensions.")
    net = model.model.to(model.device).eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), model.args.val_batch_size):
            batch = torch.as_tensor(values[start:start + model.args.val_batch_size],
                                    dtype=torch.float32, device=model.device)
            categorical = torch.cat([torch.zeros((len(batch), 1), device=model.device),
                                     batch[:, CAT_INDICES]], dim=1).long()
            cat_emb = net.embeds(categorical + net.categories_offset)
            cont_emb = torch.stack([net.simple_MLP[i](batch[:, i:i + 1])
                                    for i in range(7)], dim=1)
            embedded = torch.cat([cat_emb, cont_emb], dim=1)
            embedded += net.pos_encodings(torch.arange(embedded.shape[1], device=model.device))
            outputs.append(net(embedded[:, :net.num_categories],
                               embedded[:, net.num_categories:]).cpu().numpy().reshape(-1))
    return np.concatenate(outputs)


def predict(cli):
    directory = cli.artifact_root / cli.outcome
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(cli.input)
    if not cli.preprocessed_input:
        for column in FEATURES[7:]:
            frame[column] = frame[column].fillna("Missing").astype(str)
    values = frame[FEATURES].to_numpy()
    scores = []
    for fold in range(5):
        dimensions = metadata.get("fold_categorical_dimensions", [metadata["categorical_dimensions"]] * 5)[fold]
        model = make_model(metadata, cli.use_gpu, metadata["prediction_batch_size"], dimensions)
        model.model.load_state_dict(torch.load(directory / "models" / f"fold_{fold}.pt",
                                              map_location="cpu", weights_only=True), strict=True)
        if cli.preprocessed_input:
            transformed = values
        else:
            preprocessor = directory / "models" / f"preprocessor_fold_{fold}.pkl"
            if not preprocessor.exists():
                raise FileNotFoundError("Original preprocessing is not supplied. Use identically "
                                        "preprocessed input or a newly trained model bundle.")
            with preprocessor.open("rb") as handle:
                transformed = process_data(values, CAT_INDICES, transformer=pickle.load(handle))
        scores.append(log_risk(model, transformed))
    result = pd.DataFrame({"log_risk": np.mean(scores, axis=0)})
    if "eid" in frame:
        result.insert(0, "eid", frame["eid"].to_numpy())
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cli.output, index=False)


def train(cli):
    """Refit five folds with saved hyperparameters and fold-local preprocessing."""
    from sklearn.model_selection import StratifiedKFold, train_test_split
    metadata = json.loads((cli.artifact_root / cli.outcome / "metadata.json").read_text(encoding="utf-8"))
    if cli.output.exists():
        raise FileExistsError("Choose a new output directory; existing weights will not be overwritten.")
    data = _prepare_frame(cli.input)
    data = data.loc[pd.to_numeric(data.Ethnic, errors="coerce").isin(EUROPEAN_CODES)].copy()
    for column in FEATURES[7:]:
        data[column] = data[column].fillna("Missing").astype(str)
    x, y = data[FEATURES].to_numpy(), data[["time", "Is_Incident"]].to_numpy(dtype=float)
    x, _, y, _ = train_test_split(x, y, test_size=0.1, random_state=221, stratify=y[:, 1])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=221)
    directory = cli.output / cli.outcome
    (directory / "models").mkdir(parents=True)
    (directory / "breslow").mkdir()
    dimensions = []
    for fold, (fit, val) in enumerate(splitter.split(x, y[:, 1])):
        fitted, transformer = process_data(x[fit], CAT_INDICES, method="deep")
        validation = process_data(x[val], CAT_INDICES, transformer=transformer)
        dims = [len(categories) + 1 for categories in transformer.named_transformers_["cat"].categories_]
        dimensions.append(dims)
        model = make_model(metadata, cli.use_gpu, cat_dims=dims)
        model.args.artifact_dir = cli.output / "training_runs"
        model.args.epochs = cli.epochs
        model.fit(fitted, y[fit], validation, y[val], fold)
        torch.save(model.model.state_dict(), directory / "models" / f"fold_{fold}.pt")
        with (directory / "models" / f"preprocessor_fold_{fold}.pkl").open("wb") as handle:
            pickle.dump(transformer, handle)
        with (directory / "breslow" / f"fold_{fold}.pkl").open("wb") as handle:
            pickle.dump(model.breslow, handle)
    metadata.update(artifact="Refitted SAINT-MMP12 ensemble", categorical_dimensions=dimensions[0],
                    fold_categorical_dimensions=dimensions, preprocessing_available=True,
                    baseline_hazard_available=True, provenance="New training run, not historical weights")
    metadata.pop("files", None)
    metadata.pop("validation", None)
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "predict"])
    parser.add_argument("--outcome", choices=OUTCOMES, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/saint_mmp12"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preprocessed-input", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=1000)
    cli = parser.parse_args(argv)
    if cli.epochs < 1:
        parser.error("--epochs must be positive")
    (train if cli.stage == "train" else predict)(cli)


if __name__ == "__main__":
    main()
