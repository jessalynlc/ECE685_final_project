import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
from tqdm import tqdm

def get_predictions(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 32,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        for imgs, labs, *_ in tqdm(loader, desc=f"Eval", unit="batch"):
            imgs = imgs.to(device)
            logits, mixed_y = model(imgs)

            prob = torch.sigmoid(logits).cpu()

            preds_list.append(prob)
            labels_list.append(labs)

    preds = torch.vstack(preds_list)
    labels = torch.vstack(labels_list).type(torch.int)

    return preds, labels
