import torch.nn as nn
from models.manifold_mixup.resnet_manifold_mixup import build_model as build_manifold_mixup

def models(model_name: str) -> nn.Module:
    """
    Returns a CNN model based on the model_name.

    Args:
        model_name (str): Name of the model to return. 
        Options: 
        {
            "ManiFold_Mixup": ResNet w/ ManiFold_Mixup
        }

    Returns:
        nn.Module: An instance of the requested model.
    """
    available_models = [
        "ManiFold_Mixup"
    ]

    if  model_name == "ManiFold_Mixup":
        return build_manifold_mixup(num_classes=15)
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Available models: {available_models}")
