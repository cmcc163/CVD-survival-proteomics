"""Model registry for the models evaluated in the manuscript."""


MODEL_NAMES = (
    "LinearModel",
    "XGBoost",
    "MLP",
    "TabNet",
    "NODE",
    "FT-Transformer",
    "SAINT",
    "TabPFN",
)


def str2model(name):
    """Resolve a public model name to its implementation class."""

    if name == "LinearModel":
        from models.baseline_models import LinearModel
        return LinearModel
    if name == "XGBoost":
        from models.tree_models import XGBoost
        return XGBoost
    if name in {"MLP", "TabPFN"}:
        from models.mlp import MLP
        return MLP
    if name == "TabNet":
        from models.tabnet import TabNet
        return TabNet
    if name == "NODE":
        from models.node import NODE
        return NODE
    if name == "FT-Transformer":
        from models.ft_transformer import FTTransformer
        return FTTransformer
    if name == "SAINT":
        from models.saint import SAINT
        return SAINT
    raise ValueError(f"Unsupported manuscript model: {name}")

