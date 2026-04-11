import torch
import torch.nn as nn


class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin

    def forward(self, preds, targets):
        """
        preds:   [B]
        targets: [B]
        """
        # pairwise difference
        pred_diff = preds.unsqueeze(1) - preds.unsqueeze(0)      # [B, B]
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)

        # only keep pairs with different targets
        mask = target_diff > 0   # y_i > y_j

        if mask.sum() == 0:
            return torch.tensor(0.0, device=preds.device)

        loss = -torch.log(torch.sigmoid(pred_diff[mask] - self.margin) + 1e-8)

        return loss.mean()

class HybridLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.2):
        super().__init__()
        self.reg_loss = nn.SmoothL1Loss()
        self.rank_loss = PairwiseRankingLoss()

        self.alpha = alpha  # regression weight
        self.beta = beta    # ranking weight

    def forward(self, preds, targets):
        reg = self.reg_loss(preds, targets)
        rank = self.rank_loss(preds, targets)
        
        # 儲存各損失以便後續使用
        self.last_reg_loss = reg.item() if isinstance(reg, torch.Tensor) else reg
        self.last_rank_loss = rank.item() if isinstance(rank, torch.Tensor) else rank

        return self.alpha * reg + self.beta * rank