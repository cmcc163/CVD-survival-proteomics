"""XGBoost model used for the manuscript survival analyses."""

from __future__ import annotations

import os
import pickle

import numpy as np
import xgboost as xgb
from lifelines.utils import concordance_index

from models.basemodel import BaseModel
from utils.breslow import BreslowEstimator
from utils.io_utils import get_output_path


def c_index_metric(predictions: np.ndarray, data: xgb.DMatrix) -> tuple[str, float]:
    """Compute Harrell's C-index for XGBoost's signed Cox labels."""
    signed_time = data.get_label()
    time = np.abs(signed_time)
    event = (signed_time > 0).astype(int)
    try:
        value = concordance_index(time, -predictions, event_observed=event)
    except Exception:
        # A validation fold with no comparable pairs has an undefined C-index.
        value = 0.5
    return "c-index", float(value)


class XGBoost(BaseModel):
    """XGBoost with a Cox objective and Breslow 10-year risk calibration."""

    def __init__(self, params, args):
        super().__init__(params, args)
        self.params.update(verbosity=1, max_delta_step=1)
        if args.use_gpu:
            self.params["device"] = "cuda"

        self.objective_type = args.objective
        if args.objective == "regression":
            self.params.update(objective="reg:squarederror", eval_metric="rmse")
        elif args.objective == "classification":
            self.params.update(objective="multi:softprob", num_class=args.num_classes, eval_metric="mlogloss")
        elif args.objective == "binary":
            self.params.update(objective="binary:logistic", eval_metric="auc")
        elif args.objective == "Survival":
            self.params.update(objective="survival:cox", disable_default_eval_metric=1)
            self.breslow = BreslowEstimator()
        else:
            raise ValueError(f"Unsupported objective: {args.objective}")

    @staticmethod
    def _prepare_survival_label(y):
        """Encode events as positive times and censoring as negative times."""
        if y is None:
            return None
        y = np.asarray(y)
        return np.where(y[:, 1] == 1, y[:, 0], -y[:, 0])

    def _fit_breslow(self, matrix: xgb.DMatrix, y: np.ndarray) -> None:
        log_risk = self.model.predict(matrix, output_margin=True)
        self.breslow.fit(log_risk, y[:, 1], y[:, 0])

    def fit(self, X, y, X_val=None, y_val=None, cross_val_index=None):
        """Fit with early stopping and calibrate the baseline hazard."""
        train_label = self._prepare_survival_label(y) if self.objective_type == "Survival" else y
        valid_label = self._prepare_survival_label(y_val) if self.objective_type == "Survival" else y_val
        train = xgb.DMatrix(X, label=train_label)
        evaluations = []
        if X_val is not None and valid_label is not None:
            evaluations.append((xgb.DMatrix(X_val, label=valid_label), "eval"))

        is_survival = self.objective_type == "Survival"
        self.model = xgb.train(
            self.params,
            train,
            num_boost_round=self.args.epochs,
            evals=evaluations,
            early_stopping_rounds=self.args.early_stopping_rounds,
            verbose_eval=self.args.logging_period,
            custom_metric=c_index_metric if is_survival else None,
            maximize=is_survival,
        )
        if is_survival:
            self._fit_breslow(train, np.asarray(y))
        best_iteration = getattr(self.model, "best_iteration", self.args.epochs - 1)
        return [], [], best_iteration

    def fit_best_n_epochs(self, X, y, best_n_epochs):
        """Refit on all development data for the selected number of rounds."""
        train_label = self._prepare_survival_label(y) if self.objective_type == "Survival" else y
        train = xgb.DMatrix(X, label=train_label)
        is_survival = self.objective_type == "Survival"
        self.model = xgb.train(
            self.params,
            train,
            num_boost_round=int(best_n_epochs),
            evals=[],
            verbose_eval=self.args.logging_period,
            custom_metric=c_index_metric if is_survival else None,
            maximize=is_survival,
        )
        if is_survival:
            self._fit_breslow(train, np.asarray(y))

    def load_model(self, filename_extension=""):
        """Load a fold-specific serialized booster."""
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
        """Restore baseline-hazard calibration after loading a booster."""
        if self.objective_type == "Survival":
            self._fit_breslow(xgb.DMatrix(X_train), np.asarray(y_train))

    def predict(self, X: np.ndarray):
        """Return risk scores and, for survival, 10-year event probabilities."""
        matrix = xgb.DMatrix(X)
        if self.objective_type == "Survival":
            log_risk = self.model.predict(matrix, output_margin=True)
            survival = self.breslow.predict_survival_probabilities(log_risk, 3652.0).ravel()
            self.predictions = log_risk
            self.prediction_probabilities = 1.0 - survival
            return self.predictions, self.prediction_probabilities

        prediction = self.model.predict(matrix)
        if self.params["objective"] == "multi:softprob":
            self.prediction_probabilities = prediction
            self.predictions = np.argmax(prediction, axis=1)
        else:
            self.predictions = prediction
        return self.predictions

    def predict_proba(self, X):
        """Return class probabilities for non-survival objectives."""
        probabilities = self.model.predict(xgb.DMatrix(X))
        if self.args.objective == "binary":
            probabilities = probabilities.reshape(-1, 1)
            probabilities = np.concatenate((1 - probabilities, probabilities), axis=1)
        self.prediction_probabilities = probabilities
        return probabilities

    @classmethod
    def define_trial_parameters(cls, trial, args):
        """Define the Optuna search space used for manuscript training."""
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 0, 7),
            "eta": trial.suggest_float("eta", 1e-4, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "gamma": trial.suggest_float("gamma", 0, 5.0),
            "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-5, 10.0, log=True),
        }
