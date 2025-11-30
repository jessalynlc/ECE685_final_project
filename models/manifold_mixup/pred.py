import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import Tuple

def get_predictions(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 32,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect model outputs and labels from a dataset.

    Args:
        model (nn.Module): Trained PyTorch model used for inference.
        dataset (Dataset): Dataset that yields (image, label, ...) tuples.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 32.
        device (str, optional): Device to run inference on ("cpu" or "cuda"). Defaults to "cpu".

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            preds: numpy array of predicted probabilities with dtype np.float32, shape (N, C).
            labels: numpy array of ground-truth labels with dtype np.int64, shape (N, C).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()

    preds_list = []
    labels_list = []

    with torch.no_grad():
        for imgs, labs, *_ in loader:
            imgs = imgs.to(device)
            logits, mixed_y = model(imgs)

            prob = torch.sigmoid(logits).cpu().numpy()

            # ensure labels are numpy
            labs_np = labs.cpu().numpy()

            preds_list.append(prob)
            labels_list.append(labs_np)

    preds = np.vstack(preds_list).astype(np.float32)
    labels = np.vstack(labels_list).astype(np.int64)

    return preds, labels
