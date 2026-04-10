import pandas as pd
from sklearn.metrics import f1_score

def evaluate_participant_macro_f1(participant_predictions, golden_labels):
    """
    Compute Macro-F1 for one participant.

    Parameters
    ----------
    participant_predictions : list[str] or pd.Series
        Participant labels, e.g. ["A", "B", "Uncertain", ...]
    golden_labels : list[str] or pd.Series
        Ground-truth labels, e.g. golden_df["golden_product"]

    Returns
    -------
    float
        Macro-F1 score
    """
    y_pred = pd.Series(participant_predictions).astype(str).str.strip().reset_index(drop=True)
    y_true = pd.Series(golden_labels).astype(str).str.strip().reset_index(drop=True)

    if len(y_pred) != len(y_true):
        raise ValueError(f"Length mismatch: {len(y_pred)} predictions vs {len(y_true)} gold labels")

    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# -----------------------------------------
# example usage with golden_df
# -----------------------------------------
# participant_predictions = ["A", "B", "Uncertain", ...]
# macro_f1 = evaluate_participant_macro_f1(
#     participant_predictions,
#     golden_df["golden_product"]
# )
# print("Participant Macro-F1:", macro_f1)