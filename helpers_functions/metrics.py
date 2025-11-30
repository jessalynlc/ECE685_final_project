import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from typing import Tuple, Dict, Any


def compute_general_metrics(
    preds: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """Compute general (dataset-level) AP, AUROC, and F1-score.

    Args:
        preds (np.ndarray): Predicted probabilities, shape (N, C). Expected dtype: numeric.
        labels (np.ndarray): Ground-truth binary labels, shape (N, C). Expected dtype: integer/bool.

    Returns:
        Dict[str, float]: Dictionary with keys "AP", "AUROC", "F1-score" and float values.
    """
    # Defensive dtype casting to avoid float16 -> Python float conversion errors in sklearn
    preds = preds.astype(np.float64, copy=False)   # use float64 for maximum sklearn compatibility
    labels = labels.astype(np.int64, copy=False)

    # Binarize for F1 calculation (threshold 0.5)
    binary = (preds >= 0.5).astype(int)

    # Average Precision (AP) = area under precision-recall curve
    try:
        ap = average_precision_score(labels, preds, average="macro")
    except ValueError:
        ap = float("nan")

    # AUROC (macro)
    try:
        auroc = roc_auc_score(labels, preds, average="macro")
    except ValueError:
        auroc = float("nan")

    # Macro F1
    f1 = f1_score(labels, binary, average="macro", zero_division=0)

    return {"AP": float(ap), "AUROC": float(auroc), "F1-score": float(f1)}


def compute_classwise_metrics(
    preds: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Compute per-class AP, AUROC, and F1-score.

    Args:
        preds (np.ndarray): Predicted probabilities, shape (N, C).
        labels (np.ndarray): Ground-truth binary labels, shape (N, C).

    Returns:
        Dict[str, Dict[str, float]]: Mapping from class label string (e.g. "class_0") to metrics dict.
    """
    preds = preds.astype(np.float64, copy=False)
    labels = labels.astype(np.int64, copy=False)

    num_classes = labels.shape[1]
    binary = (preds >= 0.5).astype(int)

    class_metrics: Dict[str, Dict[str, float]] = {}

    for c in range(num_classes):
        y_true = labels[:, c]
        y_prob = preds[:, c]
        y_pred = binary[:, c]

        try:
            ap = average_precision_score(y_true, y_prob)
        except ValueError:
            ap = float("nan")

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")

        f1 = f1_score(y_true, y_pred, zero_division=0)

        class_metrics[f"class_{c}"] = {
            "AP": float(ap),
            "AUROC": float(auc),
            "F1-score": float(f1)
        }

    return class_metrics


def evaluate(preds: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Evaluate a trained model on a dataset and return metrics in a DataFrame.

    Args:
        model (nn.Module): Trained PyTorch model to evaluate.
        dataset (Dataset): Dataset that yields (image, label, ...) tuples.
        batch_size (int, optional): Batch size used for evaluation. Defaults to 32.
        device (str, optional): Device to run evaluation on. Defaults to "cpu".

    Returns:
        pd.DataFrame: DataFrame with rows ['general', 'class_wise'] and one column 'metrics'
                      where 'metrics' contains dicts of computed results.
    """
    general = compute_general_metrics(preds, labels)
    classwise = compute_classwise_metrics(preds, labels)

    df = pd.DataFrame({
        "metrics": {
            "general": general,
            "class_wise": classwise
        }
    })

    return df
