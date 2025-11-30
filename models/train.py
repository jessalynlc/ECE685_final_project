import torch.optim as optim
import torch
import torch.nn as nn
import os
from models.manifold_mixup.train import train as train_manifold_mixup
from models.mo_ex.train import train as train_mo_ex

def train(
        model_name: str, 
        model: nn.Module,
        optimizer: optim.Optimizer,
        dataset: torch.utils.data.Dataset, start_epoch: int, end_epoch: int,
        batch_size: int = 16, device: str = "cpu", 
        save: bool = False, save_every: int = 1, checkpoint_dir="checkpoints"
    ):
    """
    Train a PyTorch model on the given dataset
    Returns a CNN model based on the model_name.

    Args:
        model_name (str): Name of the model to return. 
        Options: 
        {
            "ManiFold_Mixup": ResNet w/ ManiFold_Mixup
        }
        
        model (nn.Module): PyTorch model to train.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        dataset (torch.utils.data.Dataset): Dataset to use for training.
        start_epoch (int): Starting epoch number.
        end_epoch (int): Ending epoch number (exclusive).
        batch_size (int, optional): Batch size. Defaults to 16.
        device (str, optional): Device to use ("cpu" or "cuda"). Defaults to "cpu".
        save (bool, optional): Whether to save checkpoints. Defaults to False.
        save_every (int, optional): Save every N epochs if save=True. Defaults to 1.
        checkpoint_dir (str, optional): Directory to save checkpoints. Defaults to "checkpoints".
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    available_models = [
        "ManiFold_Mixup"
    ]

    if  model_name == "ManiFold_Mixup":
        train_manifold_mixup(
            model = model,
            optimizer = optimizer,
            dataset=dataset, start_epoch=start_epoch, end_epoch=end_epoch,
            batch_size=batch_size, device=device, 
            save=save, save_every=save_every, checkpoint_dir=checkpoint_dir
        )
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Available models: {available_models}")

    
    
    