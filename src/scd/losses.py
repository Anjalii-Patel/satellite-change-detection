import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, logits, targets):
        """
        logits: [B, 1, H, W]
        targets: [B, H, W] or [B, 1, H, W]
        """
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
            
        # BCE
        bce = self.bce_loss(logits, targets)
        
        # Dice
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice_score.mean()
        
        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss
