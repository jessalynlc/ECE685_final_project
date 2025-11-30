import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Asymmetric Loss for multi-label classification (https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html).
Works with raw logits (not probabilities). Targets should be a float tensor of shape (B, C) with values in {0,1} or in [0,1] for soft labels.
"""
class AsymmetricLoss(nn.Module):
    """
    Recommended starting hyperparams from literature:
        gamma_pos = 0.0
        gamma_neg = 4.0 (strong focusing on negatives)
        clip = 0.05 (ignore very easy negatives)
    """

    def __init__(self,
                 gamma_pos: float = 0.0,
                 gamma_neg: float = 4.0,
                 clip: float = 0.05,
                 eps: float = 1e-8,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip) if clip is not None else None
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C) - raw outputs
        targets: (B, C) - float binary / soft targets
        """
        # Convert to float to avoid integer issues
        targets = targets.type_as(logits)

        # Sigmoid probabilities
        probs = torch.sigmoid(logits)

        # p_t for positives is p, for negatives it's 1-p
        pos_prob = probs
        neg_prob = 1.0 - probs

        # Clip negative probabilities. This prevents extremely
        # small negative-loss values from dominating the gradient when
        # negatives are trivial (helpful long-tailed data).
        if self.clip is not None and self.clip > 0:
            neg_prob = (neg_prob + self.clip).clamp(max=1.0)

        # Basic binary cross-entropy per element (no reduction)
        pos_loss = -targets * torch.log(pos_prob.clamp(min=self.eps))
        neg_loss = -(1.0 - targets) * torch.log(neg_prob.clamp(min=self.eps))
        loss = pos_loss + neg_loss  # shape (B, C)

        # Apply asymmetric focusing: different focusing for pos and neg
        # gamma_pos applies to positive samples, gamma_neg to negative ones.
        if (self.gamma_pos > 0.0) or (self.gamma_neg > 0.0):
            # pt = p for positives, pt = 1-p for negatives
            pt = targets * pos_prob + (1.0 - targets) * neg_prob
            # gamma per element depending on target value
            gamma = targets * self.gamma_pos + (1.0 - targets) * self.gamma_neg
            # focal factor (1 - pt)^gamma
            focal_factor = (1.0 - pt).pow(gamma)
            loss = focal_factor * loss

        # Final reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # (B, C) - no reduction
