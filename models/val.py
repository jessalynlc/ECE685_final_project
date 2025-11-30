import torch
import torch.nn as nn
from models.manifold_mixup.val import val as val_manifold_mixup

def val(
        model_name: str,
        model: nn.Module, 
        dataset: torch.utils.data.Dataset, 
        epoch: int, 
        batch_size: int = 16, 
        device: str = "cpu"
    ):
    """
    Validate a PyTorch model on the given dataset.

    Args:
        model (nn.Module): PyTorch model to validate.
        dataset (torch.utils.data.Dataset): Dataset to use for validation.
        epoch (int): Current epoch number (for logging).
        batch_size (int, optional): Batch size. Defaults to 16.
        device (str, optional): Device to use ("cpu" or "cuda"). Defaults to "cpu".
    """
    available_models = [
        "ManiFold_Mixup"
    ]

    if  model_name == "ManiFold_Mixup":
        val_manifold_mixup(
            model = model,
            dataset=dataset, epoch=epoch,
            batch_size=batch_size, device=device
        )
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Available models: {available_models}")
