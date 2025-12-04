import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from typing import Optional, Tuple, List
import torchvision.models as models



def moex(x, swap_index, norm_type, epsilon=1e-5, positive_only=False):
    '''MoEx operation'''
    dtype = x.dtype
    x = x.float()

    B, C, H, W = x.shape
    if norm_type == 'bn':
        norm_dims = [0, 2, 3]
    elif norm_type == 'in':
        norm_dims = [2, 3]
    elif norm_type == 'ln':
        norm_dims = [1, 2, 3]
    elif norm_type == 'pono':
        norm_dims = [1]
    elif norm_type.startswith('gn'):
        if norm_type.startswith('gn-d'):
            # gn-d4 means GN where each group has 4 dims
            G_dim = int(norm_type[4:])
            G = C // G_dim
        else:
            # gn4 means GN with 4 groups
            G = int(norm_type[2:])
            G_dim = C // G
        x = x.view(B, G, G_dim, H, W)
        norm_dims = [2, 3, 4]
    elif norm_type.startswith('gpono'):
        if norm_type.startswith('gpono-d'):
            # gpono-d4 means GPONO where each group has 4 dims
            G_dim = int(norm_type[len('gpono-d'):])
            G = C // G_dim
        else:
            # gpono4 means GPONO with 4 groups
            G = int(norm_type[len('gpono'):])
            G_dim = C // G
        x = x.view(B, G, G_dim, H, W)
        norm_dims = [2]
    else:
        raise NotImplementedError(f'norm_type={norm_type}')

    if positive_only:
        x_pos = F.relu(x)
        s1 = x_pos.sum(dim=norm_dims, keepdim=True)
        s2 = x_pos.pow(2).sum(dim=norm_dims, keepdim=True)
        count = x_pos.gt(0).sum(dim=norm_dims, keepdim=True)
        count[count == 0] = 1  # deal with 0/0
        mean = s1 / count
        var = s2 / count - mean.pow(2)
        std = var.add(epsilon).sqrt()
    else:
        mean = x.mean(dim=norm_dims, keepdim=True)
        std = x.var(dim=norm_dims, keepdim=True).add(epsilon).sqrt()
    swap_mean = mean[swap_index]
    swap_std = std[swap_index]
    # output = (x - mean) / std * swap_std + swap_mean
    # equvalent but for efficient
    scale = swap_std / std
    shift = swap_mean - mean * scale
    output = x * scale + shift
    return output.view(B, C, H, W).to(dtype)


class MoExResNet(nn.Module):

    def __init__(self, base_resnet: nn.Module, moex_layers: Optional[List[str]] = None):
        super().__init__()
        self.backbone = base_resnet
        self.moex_layers = moex_layers or ["stem", "C2", "C3", "C4", "C5"]

        # expose ResNet components for manual forward pass
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4
        self.avgpool = self.backbone.avgpool
        self.fc = self.backbone.fc

    def forward(self, x, swap_index=None, moex_norm='pono', moex_epsilon=1e-5,
                moex_layer='stem', training_mix: bool = True, moex_positive_only=False):
        
        moex_layer = None
        if self.training and training_mix:
            moex_layer = random.choice(self.moex_layers)
            if swap_index is None:
                B = x.size(0)
                swap_index = torch.randperm(B, device=x.device)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if swap_index is not None and moex_layer == 'stem':
            x = moex(x, swap_index, moex_norm, moex_epsilon, moex_positive_only)

        x = self.layer1(x)
        if swap_index is not None and moex_layer == 'C2':
            x = moex(x, swap_index, moex_norm, moex_epsilon, moex_positive_only)
        x = self.layer2(x)
        if swap_index is not None and moex_layer == 'C3':
            x = moex(x, swap_index, moex_norm, moex_epsilon, moex_positive_only)
        x = self.layer3(x)
        if swap_index is not None and moex_layer == 'C4':
            x = moex(x, swap_index, moex_norm, moex_epsilon, moex_positive_only)
        x = self.layer4(x)
        if swap_index is not None and moex_layer == 'C5':
            x = moex(x, swap_index, moex_norm, moex_epsilon, moex_positive_only)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def build_model(num_classes: int = 15, backbone_name: str = "resnet18"):
    """
    Creates ResNet model with mo_ex.

    args:
        num_classes: number of output labels
        backbone_name: torchvision model name
        alpha: beta distribution parameter for mixup
    
    returns:
        model: MoExResNet instance
    """

    #load a torchvision resnet
    backbone = getattr(models, backbone_name)(pretrained=False)
    #replace final FC layer to match the number of classes
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)
    #Wrap with mo_ex behavior
    model = MoExResNet(backbone)
    return model