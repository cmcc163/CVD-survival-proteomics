from pathlib import Path
import math
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


OUTCOMES = ["Total_CVD", "ASCVD", "HF"]
OUTCOME_LABELS = {"Total_CVD": "Total CVD", "ASCVD": "ASCVD", "HF": "HF"}
MODELS = ["FT-Transformer", "MLP", "NODE", "SAINT", "TabNet", "TabPFN", "XGBoost"]

RAW_BASE = Path(os.environ.get("CVD_SHAP_ROOT", "artifacts/shap"))

ROOT = Path.cwd()
OUT_DIR = Path(os.environ.get("CVD_INTERACTION_OUTPUT", "results/interactions"))
CONSENSUS_FILE = Path(os.environ.get("CVD_CONSENSUS_SHAP_FILE", "results/interpretation/consensus_shap.xlsx"))


def read_feature_metadata():
    metadata = {}
    if not CONSENSUS_FILE.exists():
        return metadata
    for outcome in OUTCOMES:
        df = pd.read_excel(CONSENSUS_FILE, sheet_name=outcome)
        mapping = {}
        for row in df.to_dict("records"):
            feature = str(row["Feature"])
            display = str(row["DisplayFeature"]).replace("\n", " ")
            mapping[feature] = {
                "display": display,
                "type": row["FeatureType"],
                "combined_rank": row.get("Combined_model_rank", np.nan),
                "combined_shap_pct": row.get("Combined_model_SHAP_pct", np.nan),
            }
        metadata[outcome] = mapping
    return metadata


def classify_pair(feature_a, feature_b, meta):
    type_a = meta.get(feature_a, {}).get("type", "Unknown")
    type_b = meta.get(feature_b, {}).get("type", "Unknown")
    protein_types = {"Protein marker", "Protein biomarker"}
    if type_a == type_b:
        if type_a == "Clinical variable":
            return "Clinical-clinical"
        if type_a in protein_types:
            return "Protein-protein"
    if "Clinical variable" in {type_a, type_b} and ({type_a, type_b} & protein_types):
        return "Clinical-protein"
    return "Other"


def display_pair(feature_a, feature_b, meta):
    label_a = meta.get(feature_a, {}).get("display", feature_a)
    label_b = meta.get(feature_b, {}).get("display", feature_b)
    return f"{label_a} x {label_b}"


def extract_model_pairs(outcome, model, meta):
    path = RAW_BASE / outcome / "all_lassonet" / model / "SHAP_Raw_Data.xlsx"
    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_excel(path, sheet_name="Mean_Abs_Interaction")
    features = raw.iloc[:, 0].astype(str).tolist()
    values = raw.iloc[:, 1:].to_numpy(dtype=float)
    if values.shape[0] != values.shape[1]:
        raise ValueError(f"Interaction matrix is not square: {path}")

    rows = []
    diagonal_sum = float(np.nansum(np.diag(values)))
    offdiag_total_double = float(np.nansum(values) - diagonal_sum)
    total_pair_interaction = offdiag_total_double / 2.0

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            val = float(np.nanmean([values[i, j], values[j, i]]))
            rows.append(
                {
                    "Outcome": outcome,
                    "OutcomeLabel": OUTCOME_LABELS[outcome],
                    "Model": model,
                    "Feature_A": features[i],
                    "Feature_B": features[j],
                    "PairKey": "||".join(sorted([features[i], features[j]])),
                    "Pair": display_pair(features[i], features[j], meta),
                    "PairType": classify_pair(features[i], features[j], meta),
                    "MeanAbsInteraction": val,
                    "TotalPairInteraction": total_pair_interaction,
                    "InteractionSharePct": 100.0 * val / total_pair_interaction
                    if total_pair_interaction > 0
                    else np.nan,
                    "DiagonalMainEffectSum": diagonal_sum,
                    "OffDiagonalInteractionSum": total_pair_interaction,
                }
            )
    df = pd.DataFrame(rows)
    df["ModelRank"] = df["MeanAbsInteraction"].rank(method="min", ascending=False).astype(int)
    df["ModelPercentile"] = 100.0 * (1.0 - (df["ModelRank"] - 1) / max(len(df) - 1, 1))
    df["Top20InModel"] = df["ModelRank"] <= 20
    df["Top50InModel"] = df["ModelRank"] <= 50
    return df


def aggregate_pairs(detail):
    group_cols = ["Outcome", "OutcomeLabel", "PairKey", "Pair", "PairType", "Feature_A", "Feature_B"]
    agg = (
        detail.groupby(group_cols, as_index=False)
        .agg(
            MeanInteractionSharePct=("InteractionSharePct", "mean"),
            MedianInteractionSharePct=("InteractionSharePct", "median"),
            SDInteractionSharePct=("InteractionSharePct", "std"),
            MeanRawInteraction=("MeanAbsInteraction", "mean"),
            MedianModelRank=("ModelRank", "median"),
            BestModelRank=("ModelRank", "min"),
            Top20ModelSupport=("Top20InModel", "sum"),
            Top50ModelSupport=("Top50InModel", "sum"),
            ModelsAvailable=("Model", "nunique"),
        )
        .sort_values(
            ["Outcome", "Top20ModelSupport", "MeanInteractionSharePct", "MedianModelRank"],
            ascending=[True, False, False, True],
        )
    )
    agg["ConsensusRank"] = (
        agg.groupby("Outcome")["MeanInteractionSharePct"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    agg["RobustCandidate"] = (agg["Top20ModelSupport"] >= 2) | (agg["Top50ModelSupport"] >= 4)
    return agg.sort_values(["Outcome", "ConsensusRank"])


def summarize_categories(detail, consensus):
    model_summary = (
        detail.groupby(["Outcome", "OutcomeLabel", "Model", "PairType"], as_index=False)
        .agg(
            PairCount=("PairKey", "nunique"),
            InteractionSharePct=("InteractionSharePct", "sum"),
            MedianPairSharePct=("InteractionSharePct", "median"),
        )
        .sort_values(["Outcome", "Model", "PairType"])
    )
    top_summary = (
        consensus[consensus["ConsensusRank"] <= 50]
        .groupby(["Outcome", "OutcomeLabel", "PairType"], as_index=False)
        .agg(
            Top50ConsensusPairs=("PairKey", "nunique"),
            MeanInteractionSharePct=("MeanInteractionSharePct", "mean"),
            RobustCandidates=("RobustCandidate", "sum"),
        )
        .sort_values(["Outcome", "PairType"])
    )
    return model_summary, top_summary


def wrap_labels(labels, width=34):
    wrapped = []
    for text in labels:
        text = str(text)
        if len(text) <= width:
            wrapped.append(text)
            continue
        parts = text.split(" x ")
        if len(parts) == 2:
            wrapped.append(parts[0] + "\n x " + parts[1])
        else:
            wrapped.append(text[:width] + "\n" + text[width:])
    return wrapped


def plot_heatmap(detail, consensus):
    top = (
        consensus.sort_values(["Outcome", "ConsensusRank"])
        .groupby("Outcome")
        .head(25)[["Outcome", "PairKey", "Pair", "ConsensusRank"]]
    )
    plot_df = detail.merge(top, on=["Outcome", "PairKey", "Pair"], how="inner")
    cmap = LinearSegmentedColormap.from_list(
        "interaction", ["#f7f7f2", "#bdd7c9", "#4b9a8d", "#243b55"]
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 10), constrained_layout=True)
    vmax = np.nanpercentile(plot_df["InteractionSharePct"], 98)
    for ax, outcome in zip(axes, OUTCOMES):
        sub = plot_df[plot_df["Outcome"] == outcome].copy()
        order = (
            sub[["Pair", "ConsensusRank"]]
            .drop_duplicates()
            .sort_values("ConsensusRank")["Pair"]
            .tolist()
        )
        matrix = (
            sub.pivot_table(index="Pair", columns="Model", values="InteractionSharePct", aggfunc="mean")
            .reindex(index=order, columns=MODELS)
        )
        im = ax.imshow(matrix.values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=13, fontweight="bold")
        ax.set_xticks(np.arange(len(MODELS)))
        ax.set_xticklabels(MODELS, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels(wrap_labels(matrix.index), fontsize=7)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.72, pad=0.01)
    cbar.set_label("Within-model share of total pairwise interaction (%)", fontsize=10)
    fig.suptitle("Top candidate predictor interactions across seven all_lassonet models", fontsize=15)
    fig.savefig(OUT_DIR / "Figure_candidate_interaction_heatmap.png", dpi=300)
    plt.close(fig)


def plot_bubble(consensus):
    top = consensus.sort_values(["Outcome", "ConsensusRank"]).groupby("Outcome").head(15)
    colors = {
        "Clinical-clinical": "#517da2",
        "Clinical-protein": "#c65b4b",
        "Protein-protein": "#559b72",
        "Other": "#777777",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    for ax, outcome in zip(axes, OUTCOMES):
        sub = top[top["Outcome"] == outcome].copy().sort_values("ConsensusRank", ascending=False)
        y = np.arange(len(sub))
        size = 35 + sub["Top20ModelSupport"].to_numpy() * 55
        ax.scatter(
            sub["MeanInteractionSharePct"],
            y,
            s=size,
            c=[colors.get(x, "#777777") for x in sub["PairType"]],
            alpha=0.88,
            edgecolor="#222222",
            linewidth=0.4,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(wrap_labels(sub["Pair"], 38), fontsize=8)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean interaction share across models (%)", fontsize=10)
        ax.grid(axis="x", color="#dddddd", linewidth=0.8)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=k, markerfacecolor=v, markersize=8)
        for k, v in colors.items()
        if k in set(top["PairType"])
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Consensus interaction magnitude and model support", fontsize=15)
    fig.savefig(OUT_DIR / "Figure_candidate_interaction_bubble.png", dpi=300)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = read_feature_metadata()
    all_details = []
    for outcome in OUTCOMES:
        for model in MODELS:
            all_details.append(extract_model_pairs(outcome, model, metadata.get(outcome, {})))
    detail = pd.concat(all_details, ignore_index=True)
    consensus = aggregate_pairs(detail)
    category_model_summary, top_category_summary = summarize_categories(detail, consensus)

    output_xlsx = OUT_DIR / "Candidate_predictor_interactions_all_lassonet.xlsx"
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        detail.to_excel(writer, sheet_name="model_pair_details", index=False)
        consensus.to_excel(writer, sheet_name="consensus_interactions", index=False)
        (
            consensus.sort_values(["Outcome", "ConsensusRank"])
            .groupby("Outcome")
            .head(50)
            .to_excel(writer, sheet_name="top_consensus_interactions", index=False)
        )
        (
            consensus[consensus["RobustCandidate"]]
            .sort_values(["Outcome", "ConsensusRank"])
            .to_excel(writer, sheet_name="robust_candidate_interactions", index=False)
        )
        category_model_summary.to_excel(writer, sheet_name="pair_type_model_summary", index=False)
        top_category_summary.to_excel(writer, sheet_name="top50_pair_type_summary", index=False)

    print(f"Wrote {output_xlsx}")
    print(f"Rows in model_pair_details: {len(detail):,}")
    print(f"Rows in consensus_interactions: {len(consensus):,}")
    print("Run plot_candidate_interactions.R to regenerate the publication-style R figures.")


if __name__ == "__main__":
    main()
