"""Fold-local preprocessing and the refitted Cox reference model."""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from models.basemodel import BaseModel
from utils.breslow import BreslowEstimator
from utils.io_utils import get_output_path


def _shift_deep_categories(values: np.ndarray, transformer: ColumnTransformer) -> np.ndarray:
    """Reserve category zero for values unseen in a training fold."""
    categorical = transformer.named_transformers_.get("cat")
    if isinstance(categorical, OrdinalEncoder) and len(categorical.categories_) > 0:
        values = np.asarray(values).copy()
        values[:, -len(categorical.categories_) :] += 1
    return values


def process_data(X, cat_index, transformer=None, method="linear"):
    """Fit on a training partition or apply an already fitted transformer.

    Linear and tree models receive one-hot categories. Neural models receive
    ordinal categories, shifted by one so unseen validation values map to zero.
    Continuous predictors are standardized in both cases.
    """
    values = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    categorical_indices = list(cat_index or [])
    numeric_indices = [index for index in range(values.shape[1]) if index not in categorical_indices]

    if transformer is None:
        transformations = [("num", StandardScaler(), numeric_indices)]
        if categorical_indices:
            if method in {"linear", "onehot"}:
                encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            elif method == "deep":
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            else:
                raise ValueError(f"Unknown preprocessing method: {method}")
            transformations.append(("cat", encoder, categorical_indices))
        transformer = ColumnTransformer(transformations, verbose_feature_names_out=False)
        transformed = transformer.fit_transform(values)
        return _shift_deep_categories(transformed, transformer), transformer

    transformed = transformer.transform(values)
    return _shift_deep_categories(transformed, transformer)


class LinearModel(BaseModel):
    """L1-regularized Cox model used as the refitted clinical reference."""

    def __init__(self, params, args):
        super().__init__(params, args)
        if args.objective != "Survival":
            raise ValueError("The publication LinearModel supports survival analysis only.")
        from sksurv.linear_model import CoxnetSurvivalAnalysis

        self.model = CoxnetSurvivalAnalysis(
            l1_ratio=1.0,
            alphas=[params.get("alpha", 0.1)],
            max_iter=5000,
            fit_baseline_model=True,
        )
        self.breslow = BreslowEstimator()

    def load_model(self, filename_extension=""):
        """Load one serialized fold-specific Cox model."""
        filename = get_output_path(
            self.args,
            directory="models",
            filename="m",
            extension=filename_extension,
            file_type="pkl",
        )
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")
        with open(filename, "rb") as handle:
            self.model = pickle.load(handle)

    def fit_breslow(self, X_train, y_train):
        """Estimate the baseline hazard using development-set predictions."""
        risk = self.model.predict(X_train)
        if isinstance(y_train, np.ndarray) and y_train.dtype.names:
            time = y_train["time"]
            event = y_train["event"].astype(int)
        else:
            time = y_train[:, 0]
            event = y_train[:, 1]
        self.breslow.fit(risk, event, time)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        """Define the Cox penalty search space."""
        return {"alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True)}
