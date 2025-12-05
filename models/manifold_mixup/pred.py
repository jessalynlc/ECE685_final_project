import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
from tqdm import tqdm

def extract_logits(model_output):
    """Model-safe logits extractor (handles tuple, list, dict, or tensor)."""
    if torch.is_tensor(model_output):
        return model_output
    if isinstance(model_output, (tuple, list)):
        return extract_logits(model_output[0])
    if isinstance(model_output, dict):
        for key in ("logits", "out", "pred", "prediction", "score"):
            if key in model_output:
                return extract_logits(model_output[key])
        for v in model_output.values():
            if torch.is_tensor(v):
                return v
    raise TypeError(f"Cannot extract logits from: {type(model_output)}")

def get_predictions(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    device: str = "cuda",
    thresholds: np.ndarray = None   # added threshold array
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect model outputs and labels from a dataset.

    Args:
        model (nn.Module): Trained PyTorch model used for inference.
        dataset (Dataset): Dataset that yields (image, label, ...) tuples.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 32.
        device (str, optional): Device to run inference on ("cpu" or "cuda"). Defaults to "cpu".

    Returns:
        probs:  float32 array of shape (N, C) with probabilities
        labels: int32 array of shape (N, C)
        preds:  int32 array of shape (N, C) with thresholded predictions if thresholds provided
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()

    preds_list = []
    probs_list = []
    labels_list = []
    preds_list = [] if thresholds is not None else None

    with torch.no_grad():
        for imgs, labs, *_ in tqdm(loader, desc="Eval", unit="batch"):
            imgs = imgs.to(device)
            raw_out = model(imgs)
            logits = extract_logits(raw_out)

            prob = torch.sigmoid(logits).cpu()    # tensor (B, C)
            probs_list.append(prob)               # <- FIXED
            labels_list.append(labs.cpu())

            if thresholds is not None:
                thr = torch.from_numpy(thresholds).float().unsqueeze(0)
                pred = (prob >= thr).int()
                preds_list.append(pred)

    probs = torch.vstack(probs_list).numpy().astype(np.float32)   # <- FIXED
    labels = torch.vstack(labels_list).numpy().astype(np.int32)

    if thresholds is not None:
        preds = torch.vstack(preds_list).numpy().astype(np.int32)
        return probs, labels, preds

    return probs, labels, None

