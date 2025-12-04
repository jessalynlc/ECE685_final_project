from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from models.asl.asl import *

def val(
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = AsymmetricLoss()
    model.to(device)
    model.eval()  # Set model to evaluation mode

    running_loss = 0.0

    with torch.no_grad():  # Disable gradient computation
        with tqdm(loader, desc=f"Validation Epoch {epoch}", unit="batch") as tepoch:
            for imgs, labels, _ in tepoch:
                imgs, labels = imgs.to(device), labels.to(device)

                logits, mixed_y = model(imgs, labels)
                loss = criterion(logits, mixed_y)

                running_loss += loss.item() * imgs.size(0)
                tepoch.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(tepoch)*batch_size
    print(f"Validation Epoch {epoch} completed - Average Loss: {epoch_loss:.4f}")
