from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import os
from tqdm import tqdm
import torch.nn as nn
from models.asl.asl import *

def train(
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
    criterion = AsymmetricLoss()
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
