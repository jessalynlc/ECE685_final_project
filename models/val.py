import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.manifold_mixup.val import val as val_manifold_mixup
from models.mo_ex.val import val as val_mo_ex
from models.asl.val import val as val_asl
from models.aug_all.val import val as val_aug_all

def val(
        model_name: str,
        model: nn.Module, 
        dataset: torch.utils.data.Dataset, 
        epoch: int, 
        batch_size: int = 16, 
        device: str = "cuda"
    ):
    """
    Validate a PyTorch model on the given dataset.

    Args:
        model_name (str): Name of the model to return. 
        Options: 
        {
            "Base",
            "ManiFold_Mixup",
            "Mo_Ex",
            "ASL",
            "All",
            "All_stochastic"
        }
        model (nn.Module): PyTorch model to validate.
        dataset (torch.utils.data.Dataset): Dataset to use for validation.
        epoch (int): Current epoch number (for logging).
        batch_size (int, optional): Batch size. Defaults to 16.
        device (str, optional): Device to use ("cpu" or "cuda"). Defaults to "cpu".
    """
    available_models = [
        "Base",
        "ManiFold_Mixup",
        "Mo_Ex",
        "ASL",
        "All",
        "All_stochastic"
    ]

    if model_name == "Base":
        val_base_model(
            model = model,
            dataset=dataset, epoch=epoch,
            batch_size=batch_size, device=device
        )
    
    elif model_name == "ManiFold_Mixup":
        val_manifold_mixup(
            model = model,
            dataset=dataset, epoch=epoch,
            batch_size=batch_size, device=device
        )
    
    elif model_name == "Mo_Ex":
        val_mo_ex(
            model = model,
            dataset=dataset, epoch=epoch,
            batch_size=batch_size, device=device
        )
    
    elif model_name == "ASL":
        val_asl(
            model = model,
            dataset=dataset, epoch=epoch,
            batch_size=batch_size, device=device
        )
    
    elif model_name == "All" or model_name == "All_stochastic":
        val_aug_all(
            model = model,
            dataset=dataset, epoch=epoch,
            batch_size=batch_size, device=device
        )
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Available models: {available_models}")


def val_base_model(
        model: nn.Module, 
        dataset: torch.utils.data.Dataset, 
        epoch: int, 
        batch_size: int = 16, 
        device: str = "cuda"
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
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    model.eval()  # Set model to evaluation mode

    running_loss = 0.0

    with torch.no_grad():  # Disable gradient computation
        with tqdm(loader, desc=f"Validation Epoch {epoch}", unit="batch") as tepoch:
            for imgs, labels, _ in tepoch:
                imgs, labels = imgs.to(device), labels.to(device)

                logits = model(imgs)
                loss = criterion(logits, labels)

                running_loss += loss.item() * imgs.size(0)
                tepoch.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(tepoch)*batch_size
    print(f"Validation Epoch {epoch} completed - Average Loss: {epoch_loss:.4f}")
