import torch
import pandas as pd
from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelAUROC,
    MultilabelF1Score,
)

def evaluate(
    model_name: str,
    model: torch.nn.Module,
    pred,          # prediction function: pred(model_name, model, dataset, batch_size)
    dataset,
    batch_size: int = 32,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluate a model using a custom prediction function with a progress bar.

    Args:
        model_name (str): Name of the model (passed to pred).
        model (nn.Module): PyTorch model.
        pred (callable): Function pred(model_name, model, dataset, batch_size) -> (preds, labels)
        dataset: Dataset to evaluate.
        batch_size (int, optional): Batch size for evaluation.
        threshold (float, optional): Threshold to binarize predicted probabilities for confusion matrix. Default 0.5

    Returns:
        pd.DataFrame: metrics ('general' and 'class_wise')
        torch.Tensor: raw confusion matrix with shape (C, 2, 2) where each per-class matrix is:
                      [[TN, FP],
                       [FN, TP]]
    """

    model.eval()
    model.to("cpu")  # pred function should handle device if needed

    # pred 
    preds, labels = pred(model_name=model_name, model=model, dataset=dataset, batch_size=batch_size)
    num_classes = preds.shape[1]


    # General metrics (macro)
    ap_general = MultilabelAveragePrecision(num_labels=num_classes, average="macro")
    auroc_general = MultilabelAUROC(num_labels=num_classes, average="macro")
    f1_general = MultilabelF1Score(num_labels=num_classes, average="macro")

    general_metrics = {
        "AP": float(ap_general(preds, labels)),
        "AUROC": float(auroc_general(preds, labels)),
        "F1-score": float(f1_general(preds, labels)),
    }

    # Class-wise metrics
    ap_classwise = MultilabelAveragePrecision(num_labels=num_classes, average=None)
    auroc_classwise = MultilabelAUROC(num_labels=num_classes, average=None)
    f1_classwise = MultilabelF1Score(num_labels=num_classes, average=None)

    ap_per_class = ap_classwise(preds, labels).tolist()
    auroc_per_class = auroc_classwise(preds, labels).tolist()
    f1_per_class = f1_classwise(preds, labels).tolist()

    class_metrics = {
        f"class_{i}": {
            "AP": float(ap_per_class[i]),
            "AUROC": float(auroc_per_class[i]),
            "F1-score": float(f1_per_class[i]),
        }
        for i in range(num_classes)
    }

    df = pd.DataFrame({
        "metrics": {
            "general": general_metrics,
            "class_wise": class_metrics
        }
    })

    # Confusion matrix (per-class)
    # Binarize predictions at the provided threshold
    preds_bin = (preds >= threshold).to(torch.int64)
    labels_bin = labels.to(torch.int64)

    # confusion shape (C, 2, 2) with [[TN, FP],[FN, TP]] per class
    confusion = torch.zeros((num_classes, 2, 2), dtype=torch.int64)

    # Find per-class TP, FP, FN, TN
    for i in range(num_classes):
        p = preds_bin[:, i]
        t = labels_bin[:, i]

        tp = int(((p == 1) & (t == 1)).sum().item())
        fp = int(((p == 1) & (t == 0)).sum().item())
        fn = int(((p == 0) & (t == 1)).sum().item())
        tn = int(((p == 0) & (t == 0)).sum().item())

        confusion[i, 0, 0] = tn
        confusion[i, 0, 1] = fp
        confusion[i, 1, 0] = fn
        confusion[i, 1, 1] = tp

    return df, confusion
