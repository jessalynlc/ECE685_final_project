import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
from models.manifold_mixup.pred import get_predictions as pred_manifold_mixup

def pred(
    model_name: str,
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 32,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect model outputs and labels from a dataset.

    Args:
        model_name (str): Name of the model to return. 
        Options: 
        {
            "ManiFold_Mixup": ResNet w/ ManiFold_Mixup
        }
        model (nn.Module): Trained PyTorch model used for inference.
        dataset (Dataset): Dataset that yields (image, label, ...) tuples.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 32.
        device (str, optional): Device to run inference on ("cpu" or "cuda"). Defaults to "cpu".

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            preds: numpy array of predicted probabilities with dtype np.float32, shape (N, C).
            labels: numpy array of ground-truth labels with dtype np.int64, shape (N, C).
    """
    available_models = [
        "ManiFold_Mixup"
    ]

    if  model_name == "ManiFold_Mixup":
        return pred_manifold_mixup(
            model=model, 
            dataset=dataset, 
            batch_size=batch_size, device=device
        )
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Available models: {available_models}")

