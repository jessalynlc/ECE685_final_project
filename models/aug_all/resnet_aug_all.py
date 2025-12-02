"""
Manifold Mixup implementation for ResNet backbones.

Designed to be imported from main.ipynb as:

from data_aug.manifold_mixup import ResNetManifoldMixup, build_model

This module exposes:
- ResNetManifoldMixup: a nn.Module wrapper that applies manifold mixup during training
- build_model: convenience builder to instantiate a model with custom num_classes and backbone

References:
- Manifold Mixup Paper: https://arxiv.org/abs/1806.0523
"""

import random
from typing import Optional, Tuple, List
import torch.nn.functional as F

import torch
import torch.nn as nn
import torchvision.models as models

# This is a helper that samples lambda from Beta(alpha, alpha)
def sample_lambda(alpha: float, device: torch.device) -> torch.Tensor:
    """
    Draws a mixup coefficient lambda from a beta distribution.
    If alpha <= 0, mixup is disabled and lambda = 1.

    args:
        alpha: float, the alpha parameter of the Beta distribution controlling mix strength
    returns:
        lam: scalar tensor in [0,1]
    """
    if alpha <= 0:
        return torch.tensor(1.0, device=device) #no mixup --> lambda = 1
    lam = torch.distributions.Beta(alpha, alpha).sample().to(device)
    return lam


class ResNetMultiMethodAugmentation(nn.Module):
    """
    This wraps a ResNet model so that data augmentation can be applied inside intermediate layers.

    args:
        base_resnet: nn.Module, torchvision ResNet model
        mix_layers: list of layer names where mixup may occur
        alpha: see above
    behavior:
        Randomly selects a mix_layer during each forward pass. And then mixes feature maps at that layer and mixes labels accordingly.
    """
    def __init__(self, base_resnet: nn.Module, mix_layers: Optional[List[str]] = None, moex_layers: Optional[List[str]] = None, alpha: float = 2.0):
        super().__init__()
        self.backbone = base_resnet
        self.mix_layers = mix_layers or ["layer1", "layer2", "layer3", "layer4", "fc"]
        self.moex_layers = moex_layers or ["stem", "C2", "C3", "C4", "C5"]
        self.alpha = alpha

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

    # forward pass with potential manifold mixup
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None,
                swap_index=None, moex_norm='pono', moex_epsilon=1e-5,
                moex_layer='stem', moex_positive_only=False,
                training_mix: bool = True, ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass that applies manifold mixup during training.
        
        args:
            x: input tensor of shape (B, C, H, W)
            y: lables (B. num_classes)
            training_mix: if False, doesn't do mixup
        returns:
            logits: model output
            mixed_y: mixed lables
        """
        device = x.device
        mix_layer = None
        if self.training and training_mix and self.alpha > 0:
            mix_layer = random.choice(self.mix_layers)
            moex_layer = random.choice(self.moex_layers)
            if swap_index is None:
                B = x.size(0)
                swap_index = torch.randperm(B, device=x.device)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        lam = None
        perm = None
        mixed_y = None

        #this function does the feature map mixing. Mixes feature maps and labels using lambda. 
        #Completed one per forward call.
        def do_mix(feat: torch.Tensor):
            nonlocal lam, perm, mixed_y
            #Initialize lambda and permuatation once
            if lam is None:
                lam = sample_lambda(self.alpha, device)
                perm = torch.randperm(feat.size(0), device=device)
                #mix labels if provided
                if y is not None:
                    y_shuf = y[perm]
                    mixed_y = lam * y + (1.0 - lam) * y_shuf
            #mix feature maps
            feat_shuf = feat[perm]
            return lam * feat + (1.0 - lam) * feat_shuf
        
        
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

        if swap_index is not None and moex_layer == 'stem':
                out = moex(out, swap_index, moex_norm, moex_epsilon, moex_positive_only)
                
        #Residual layers
        out = self.layer1(out)
        if mix_layer == "layer1":
            out = do_mix(out)
            if swap_index is not None and moex_layer == 'C2':
                out = moex(out, swap_index, moex_norm, moex_epsilon, moex_positive_only)

        out = self.layer2(out)
        if mix_layer == "layer2":
            out = do_mix(out)
            if swap_index is not None and moex_layer == 'C3':
                out = moex(out, swap_index, moex_norm, moex_epsilon, moex_positive_only)

        out = self.layer3(out)
        if mix_layer == "layer3":
            out = do_mix(out)
            if swap_index is not None and moex_layer == 'C4':
                out = moex(out, swap_index, moex_norm, moex_epsilon, moex_positive_only)

        out = self.layer4(out)
        if mix_layer == "layer4":
            out = do_mix(out)
            if swap_index is not None and moex_layer == 'C5':
                out = moex(out, swap_index, moex_norm, moex_epsilon, moex_positive_only)

        #Global pooling and FC
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        if mix_layer == "fc":
            out = do_mix(out)

        #find classifier
        logits = self.fc(out)
        return logits, mixed_y


def build_model(num_classes: int = 14, backbone_name: str = "resnet18", alpha: float = 2.0):
    """
    Creates ResNet model with multi-method augmentation.

    args:
        num_classes: number of output labels
        backbone_name: torchvision model name
        alpha: beta distribution parameter for mixup
    
    returns:
        model: ResNetManifoldMixup instance
    """

    #load a torchvision resnet
    backbone = getattr(models, backbone_name)(pretrained=False)
    #replace final FC layer to match the number of classes
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)
    #Wrap with manifold mixup behavior
    model = ResNetMultiMethodAugmentation(backbone, alpha=alpha)
    return model
