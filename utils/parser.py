"""Command-line configuration shared by all modelling stages."""

import argparse
from pathlib import Path

import configargparse


MODEL_CHOICES = [
    "LinearModel",
    "XGBoost",
    "MLP",
    "TabNet",
    "NODE",
    "FT-Transformer",
    "SAINT",
    "TabPFN",
]
OUTCOME_CHOICES = ["Total_CVD", "ASCVD", "HF"]
PREDICTOR_CHOICES = ["classic", "protein", "all"]


def get_parser() -> configargparse.ArgumentParser:
    """Return the publication pipeline parser.

    Command-line arguments override values read from the YAML configuration.
    Each invocation operates on one outcome, predictor set, and model. Batch
    execution is implemented by ``run.py batch`` rather than hard-coded loops.
    """

    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add("--config", is_config_file_arg=True, default="config/analysis.yml")
    parser.add("--dataset", default="UKB")
    parser.add("--objective", default="Survival", choices=["Survival"])
    parser.add("--disease-type", dest="disease_type", required=True, choices=OUTCOME_CHOICES)
    parser.add("--data-type", dest="data_type", required=True, choices=PREDICTOR_CHOICES)
    parser.add("--model-name", dest="model_name", required=True, choices=MODEL_CHOICES)
    parser.add("--protein-source", dest="protein_source", default="lassonet", choices=["lassonet"])

    parser.add("--path", type=Path, required=True, help="Endpoint-specific participant CSV.")
    parser.add("--protein-path", type=Path, help="LassoNet protein panel CSV or its directory.")
    parser.add("--all-proteins-path", type=Path, help="CSV containing the 2,922 protein column names.")
    parser.add("--reduced-panel-path", type=Path, help="Kneedle-selected predictor table.")
    parser.add("--output-dir", type=Path, default=Path("results"))
    parser.add("--artifact-dir", type=Path, default=Path("artifacts"))

    parser.add("--ratio", type=float, default=0.10, help="European hold-out fraction.")
    parser.add("--num-splits", type=int, default=5)
    parser.add("--n-trials", type=int, default=100)
    parser.add("--direction", choices=["maximize"], default="maximize")
    parser.add("--seed", type=int, default=221)
    parser.add("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add("--gpu-ids", type=int, nargs="+", default=[0])
    parser.add("--data-parallel", action=argparse.BooleanOptionalAction, default=False)
    parser.add("--optimize-hyperparameters", action=argparse.BooleanOptionalAction, default=True)
    parser.add("--scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add("--target-encode", action=argparse.BooleanOptionalAction, default=True)
    parser.add("--one-hot-encode", action=argparse.BooleanOptionalAction, default=False)
    parser.add("--batch-size", type=int, default=1024)
    parser.add("--val-batch-size", type=int, default=1024)
    parser.add("--early-stopping-rounds", type=int, default=20)
    parser.add("--epochs", type=int, default=1000)
    parser.add("--logging-period", type=int, default=100)
    parser.add("--bootstrap-repeats", type=int, default=1000)
    parser.add("--lassonet-repeats", type=int, default=100)
    parser.add("--lassonet-train-ratio", type=float, default=0.70)
    parser.add("--lassonet-selection-threshold", type=float, default=0.80)
    parser.add("--num-features", type=int, default=1)
    parser.add("--num-classes", type=int, default=1)
    parser.add("--cat-idx", type=int, nargs="*", default=[])
    parser.add("--cat-dims", type=int, nargs="*", default=[])
    return parser


def get_given_parameters_parser() -> configargparse.ArgumentParser:
    """Compatibility parser for modules that load an existing Optuna study."""

    parser = get_parser()
    parser.add("--best-params-file", default=None)
    parser.add("--parameters", default=None)
    return parser


def get_attribution_parser() -> configargparse.ArgumentParser:
    """Compatibility parser for SHAP stages."""

    return get_parser()
