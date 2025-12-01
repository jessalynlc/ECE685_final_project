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

# TODO: COMPLETE 
#modifying this to accept mix manifold and moex
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
    def __init__(self, base_resnet: nn.Module, mix_layers: Optional[List[str]] = None, alpha: float = 2.0,
                 mix_method: str = "both", #could be manifold, moex, or both
                 moex_prob: float = 0.5): #probability to apply MoEx when chosen
        super().__init__()
        self.backbone = base_resnet
        self.mix_layers = mix_layers or ["layer1", "layer2", "layer3", "layer4", "fc"]
        self.alpha = alpha
        self.mix_method = mix_method
        self.moex_prob = moex_prob

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
                training_mix: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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

       #MoEx behavior (swapping moments mean/std across batch and re-inject)
        def do_moex(feat: torch.Tensor):
            nonlocal lam, perm, mixed_y
            B, C = feat.shape[:2]
            # compute per-sample mean/std across spatial dims (H,W)
            # keep dims for broadcasting: shape (B,C,1,1)
            eps = 1e-6
            mu = feat.mean(dim=(2,3), keepdim=True)               # (B,C,1,1)
            sigma = feat.std(dim=(2,3), unbiased=False, keepdim=True) + eps

            # normalized features
            feat_norm = (feat - mu) / sigma

            # sample permutation & lambda (lambda used for label interpolation)
            if perm is None:
                perm = torch.randperm(B, device=device)
                # for MoEx, paper suggests high interpolation weight (e.g. lam ≈ 0.9)
                lam = sample_lambda(self.alpha, device)
                if y is not None:
                    y_shuf = y[perm]
                    mixed_y = lam * y + (1.0 - lam) * y_shuf

            mu_perm = mu[perm]
            sigma_perm = sigma[perm]

            #mix moments with original moments using lam:
            #final moments = lam * original_moments + (1-lam) * permuted_moments
            #Allows softer exchange instead of full swap
            if lam is None:
                lam = torch.tensor(1.0, device=device)
            mu_mix = lam * mu + (1.0 - lam) * mu_perm
            sigma_mix = lam * sigma + (1.0 - lam) * sigma_perm

            # re-add moments into normalized features
            feat_mixed = feat_norm * sigma_mix + mu_mix
            return feat_mixed
        
        # choose which mixing fn to call at selected layer
        def maybe_mix(feat: torch.Tensor):
            # If mix_method == both, randomly pick which method to apply each forward
            method = self.mix_method
            if self.mix_method == "both":
                method = random.choice(["manifold", "moex"])
            if method == "manifold":
                return do_mix(feat)
            elif method == "moex":
                # optionally use a probability to do MoEx (paper uses prob p)
                if random.random() <= self.moex_prob:
                    return do_moex(feat)
                else:
                    return feat
            else:
                return feat
        
        #Residual layers
        out = self.layer1(out)
        if mix_layer == "layer1":
            out = maybe_mix(out)

        out = self.layer2(out)
        if mix_layer == "layer2":
            out = maybe_mix(out)

        out = self.layer3(out)
        if mix_layer == "layer3":
            out = maybe_mix(out)

        out = self.layer4(out)
        if mix_layer == "layer4":
            out = maybe_mix(out)

        #Global pooling and FC
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        if mix_layer == "fc":
            out = maybe_mix(out)

        #find classifier
        logits = self.fc(out)
        return logits, mixed_y


def build_model(num_classes: int = 14, backbone_name: str = "resnet18", alpha: float = 2.0,
                mix_method: str = "both", moex_prob: float = 0.5):
    """
    Creates ResNet model with manifold mixup.

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
    model = ResNetMultiMethodAugmentation(backbone, alpha=alpha, mix_method="both", moex_prob=0.5)
    return model
