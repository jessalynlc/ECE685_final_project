import torch.optim as optim
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.manifold_mixup.train import train as train_manifold_mixup
from models.mo_ex.train import train as train_mo_ex
from models.asl.train import train as train_asl
from models.aug_all.train import train as train_aug_all

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
            "Base",
            "ManiFold_Mixup",
            "Mo_Ex",
            "ASL",
            "All",
            "All_stochastic"
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
        "Base",
        "ManiFold_Mixup",
        "Mo_Ex",
        "ASL",
        "All",
        "All_stochastic"
    ]

    if model_name == "Base":
        train_base_model(
            model = model,
            optimizer = optimizer,
            dataset=dataset, start_epoch=start_epoch, end_epoch=end_epoch,
            batch_size=batch_size, device=device, 
            save=save, save_every=save_every, checkpoint_dir=checkpoint_dir
        )
    
    elif model_name == "ManiFold_Mixup":
        train_manifold_mixup(
            model = model,
            optimizer = optimizer,
            dataset=dataset, start_epoch=start_epoch, end_epoch=end_epoch,
            batch_size=batch_size, device=device, 
            save=save, save_every=save_every, checkpoint_dir=checkpoint_dir
        )
    
    elif model_name == "Mo_Ex":
        train_mo_ex(
            model = model,
            optimizer = optimizer,
            dataset=dataset, start_epoch=start_epoch, end_epoch=end_epoch,
            batch_size=batch_size, device=device, 
            save=save, save_every=save_every, checkpoint_dir=checkpoint_dir
        )
    
    elif model_name == "ASL":
        train_asl(
            model = model,
            optimizer = optimizer,
            dataset=dataset, start_epoch=start_epoch, end_epoch=end_epoch,
            batch_size=batch_size, device=device, 
            save=save, save_every=save_every, checkpoint_dir=checkpoint_dir
        )
    
    elif model_name == "All" or model_name == "All_stochastic":
        train_aug_all(
            model = model,
            optimizer = optimizer,
            dataset=dataset, start_epoch=start_epoch, end_epoch=end_epoch,
            batch_size=batch_size, device=device, 
            save=save, save_every=save_every, checkpoint_dir=checkpoint_dir
        )
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Available models: {available_models}")



def train_base_model(
        model: nn.Module, 
        optimizer: optim.Optimizer,
        dataset: torch.utils.data.Dataset, start_epoch: int, end_epoch: int,
        batch_size: int = 16, device: str = "cpu", 
        save: bool = False, save_every: int = 1, checkpoint_dir="checkpoints"
    ):
    """
    Train a PyTorch model on the given dataset.

    Args:
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)

    for epoch in range(start_epoch, end_epoch):
        model.train()
        running_loss = 0.0

        # tqdm progress bar for batches
        with tqdm(loader, desc=f"Epoch {epoch+1}/{end_epoch}", unit="batch") as tepoch:
            for imgs, labels, _ in tepoch:
                imgs, labels = imgs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                tepoch.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(tepoch)*batch_size
        print(f"Epoch {epoch+1}/{end_epoch} completed - Average Loss: {epoch_loss:.4f}")

        # Save checkpoint according to save flag and save_every
        if save and ((epoch + 1) % save_every == 0):
            checkpoint_path = os.path.join(checkpoint_dir, f"clf_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    