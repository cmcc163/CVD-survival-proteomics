"""FT-Transformer survival model with feature-wise column attention."""

from models.attention_survival import AttentionSurvivalModel


class FTTransformer(AttentionSurvivalModel):
    """FT-Transformer implementation using column attention only."""

    attention_type = "col"

