"""Stable endpoint-specific protein selection with repeated-split LassoNet."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lassonet import LassoNetCoxRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models.baseline_models import process_data
from utils.load_data import load_datas


def save_frequency_results(
    counts,
    n_repeats,
    feature_names,
    output_dir,
    file_tag,
    threshold=0.8,
    feature_count_history=None,
    c_index_history=None,
):
    """Write aggregate selection frequencies; never write participant rows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_count_history = feature_count_history or []
    c_index_history = c_index_history or []

    table = pd.DataFrame(
        {
            "Protein_Name": np.asarray(feature_names)[: len(counts)],
            "Selection_Frequency": counts / n_repeats,
            "Selection_Count": counts,
        }
    )
    table["Is_Selected"] = table["Selection_Frequency"] >= threshold
    table = table.sort_values("Selection_Frequency", ascending=False)
    table.to_csv(output_dir / f"lassonet_freq_all_proteins_{file_tag}.csv", index=False)
    table.loc[table["Is_Selected"]].to_csv(
        output_dir / f"lassonet_freq_selected_threshold_{int(threshold * 100)}_{file_tag}.csv",
        index=False,
    )

    completed = min(len(feature_count_history), len(c_index_history))
    pd.DataFrame(
        {
            "Run_ID": range(1, completed + 1),
            "Best_C_Index": c_index_history[:completed],
            "Num_Features_Selected": feature_count_history[:completed],
        }
    ).to_csv(output_dir / f"lassonet_run_history_{file_tag}.csv", index=False)

    stats = {
        "requested_repeats": n_repeats,
        "completed_repeats": completed,
        "threshold": threshold,
        "n_selected": int(table["Is_Selected"].sum()),
        "mean_validation_c_index": float(np.mean(c_index_history)) if c_index_history else None,
        "mean_selected_features": float(np.mean(feature_count_history)) if feature_count_history else None,
    }
    with (output_dir / f"lassonet_freq_stats_{file_tag}.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)


def fit_repeated_split_selection_lassonet(X, y, args, num_proteins, n_repeats=100, train_ratio=0.7):
    """Count proteins selected across stratified 70:30 development splits."""
    counts = np.zeros(num_proteins, dtype=int)
    selected_counts = []
    validation_c_indices = []
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    seeds = np.random.default_rng(args.seed).integers(0, 100_000, size=n_repeats)

    for split_seed in tqdm(seeds, desc="Repeated LassoNet selection"):
        x_train, x_validation, y_train, y_validation = train_test_split(
            X,
            y,
            test_size=1 - train_ratio,
            stratify=y[:, 1],
            random_state=int(split_seed),
        )
        # The encoder is fitted on this split's training partition only.
        x_train, transformer = process_data(x_train, args.cat_idx, method="onehot")
        x_validation = process_data(x_validation, args.cat_idx, transformer=transformer)

        model = LassoNetCoxRegressor(
            hidden_dims=(1024, 128),
            path_multiplier=1.05,
            M=10,
            dropout=0.3,
            patience=(100, 20),
            n_iters=(1000, 200),
            gamma=3e-05,
            verbose=0,
            random_state=int(split_seed),
            torch_seed=int(split_seed),
            device=device,
            tie_approximation="breslow",
        )
        try:
            history = model.path(
                x_train,
                y_train,
                X_val=x_validation,
                y_val=y_validation,
                return_state_dicts=False,
                disable_lambda_warning=False,
            )
        except Exception as error:
            print(f"LassoNet split {split_seed} failed: {error}")
            continue
        if not history:
            continue

        best = max(history, key=lambda item: item.val_c_index)
        protein_mask = best.selected.cpu().numpy()[:num_proteins].astype(bool)
        counts += protein_mask.astype(int)
        selected_counts.append(int(protein_mask.sum()))
        validation_c_indices.append(float(best.val_c_index))

    return counts, selected_counts, validation_c_indices


def main(args):
    """Run selection after reserving the untouched European hold-out set."""
    x_eur, y_eur, _, _, _, _, cat_idx, feature_names = load_datas(args, data_type="all")
    args.cat_idx = cat_idx
    num_proteins = len(feature_names) - 11
    x_development, _, y_development, _ = train_test_split(
        x_eur,
        y_eur,
        test_size=args.ratio,
        stratify=y_eur[:, 1],
        random_state=args.seed,
    )
    counts, selected_counts, c_indices = fit_repeated_split_selection_lassonet(
        x_development,
        y_development,
        args=args,
        num_proteins=num_proteins,
        n_repeats=args.lassonet_repeats,
        train_ratio=args.lassonet_train_ratio,
    )
    output_dir = Path(args.output_dir) / "protein_selection" / args.disease_type
    save_frequency_results(
        counts,
        args.lassonet_repeats,
        feature_names[:num_proteins],
        output_dir,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        threshold=args.lassonet_selection_threshold,
        feature_count_history=selected_counts,
        c_index_history=c_indices,
    )
