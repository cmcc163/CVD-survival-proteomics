"""Public command-line interface for the manuscript analysis pipeline."""

from __future__ import annotations

import argparse
import sys

from utils.parser import get_parser


STAGES = (
    "select-proteins",
    "train",
    "predict",
    "oof",
    "xgboost-shap",
    "surrogate-shap",
    "tabpfn-embed",
    "tabpfn-train",
    "tabpfn-predict",
    "prevent",
    "reduced-train",
    "reduced-predict",
)


def _parse_stage(argv: list[str]):
    parser = argparse.ArgumentParser(
        add_help=False,
        description="Run one stage of the CVD survival analysis pipeline.",
    )
    parser.add_argument("stage", choices=STAGES)
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run one explicitly configured analysis stage."""

    cli_args = list(sys.argv[1:] if argv is None else argv)
    if not cli_args or cli_args == ["--help"] or cli_args == ["-h"]:
        parser = argparse.ArgumentParser(description="Run one stage of the CVD survival analysis pipeline.")
        parser.add_argument("stage", choices=STAGES)
        parser.print_help()
        return
    stage_args, remaining = _parse_stage(cli_args)
    args = get_parser().parse_args(remaining)

    if stage_args.stage == "select-proteins":
        from scripts.core.select_proteins import main as run
    elif stage_args.stage == "train":
        from scripts.core.train_models import main as run
    elif stage_args.stage == "predict":
        from scripts.core.generate_predictions import main_once as run
    elif stage_args.stage == "oof":
        from scripts.core.generate_oof_predictions import main as run
    elif stage_args.stage == "xgboost-shap":
        from scripts.core.explain_xgboost import compute_xgboost_ensemble_shap as run
    elif stage_args.stage == "surrogate-shap":
        from scripts.core.explain_surrogate import compute_unified_shap as run
    elif stage_args.stage == "tabpfn-embed":
        from scripts.core.extract_tabpfn_embeddings import generate_and_save_embeddings as run
    elif stage_args.stage == "tabpfn-train":
        from scripts.core.train_tabpfn_survival import main as run
    elif stage_args.stage == "tabpfn-predict":
        from scripts.core.predict_tabpfn_survival import run_inference_with_best_params as run
    elif stage_args.stage == "prevent":
        from scripts.core.predict_prevent import main as run
    elif stage_args.stage == "reduced-train":
        from scripts.core.train_reduced_saint import main as run
    else:
        from scripts.core.predict_reduced_saint import main_once as run

    run(args)


if __name__ == "__main__":
    main()
