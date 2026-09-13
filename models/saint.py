"""SAINT survival model with column and row attention."""

from models.attention_survival import AttentionSurvivalModel


class SAINT(AttentionSurvivalModel):
    """SAINT implementation using combined column and row attention."""

    attention_type = "colrow"

