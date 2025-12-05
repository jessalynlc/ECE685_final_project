import torch.nn as nn
import torchvision.models as torchvision_models
from torchvision.models import resnet18, ResNet18_Weights
from models.manifold_mixup.resnet_manifold_mixup import build_model as build_manifold_mixup
from models.mo_ex.resnet_mo_ex import build_model as build_mo_ex

from models.aug_all.aug_all import build_model as build_aug_all_stochastic
from models.aug_all.resnet_aug_all import build_model as build_aug_all

def models(model_name: str, backbone_name: str = "resnet18", num_classes: int = 15) -> nn.Module:
    """
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
        backbone (str): Name of the backbone model. 
        num_classes (int): Number of classes in final layer. Defaults to 15.

    Returns:
        nn.Module: An instance of the requested model.
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
        return build_base_model(num_classes=num_classes, backbone_name=backbone_name)
    
    elif model_name == "ManiFold_Mixup":
        return build_manifold_mixup(num_classes=num_classes, backbone_name=backbone_name)
    
    elif model_name == "Mo_Ex":
        return build_mo_ex(num_classes=num_classes, backbone_name=backbone_name)
    
    elif model_name == "ASL":
        return build_base_model(num_classes=num_classes, backbone_name=backbone_name)
    
    elif model_name == "All":
        return build_aug_all(num_classes=num_classes, backbone_name=backbone_name)
    
    elif model_name == "All_stochastic":
        return build_aug_all_stochastic(num_classes=num_classes, backbone_name=backbone_name)

    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Available models: {available_models}")
    

def build_base_model(num_classes: int = 15, backbone_name: str = "resnet18"):
    """
    Creates ResNet model with manifold mixup.

    args:
        num_classes: number of output labels
        backbone_name: torchvision model name
        alpha: beta distribution parameter for mixup
    
    returns:
        model: ResNetBase instance
    """

    #load a torchvision resnet
    #backbone = getattr(torchvision_models, backbone_name)(pretrained=False)
    #NEW REPLACEMENT WITH RESNET18 WITH PRETRAINED WEIGHTS
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    #replace final FC layer to match the number of classes
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)
    return backbone
