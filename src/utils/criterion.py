import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseRankingLoss(nn.Module):
    def __init__(
        self,
        margin: float = 0.0,
        min_target_diff: float = 0.0,
        weight_by_target_diff: bool = True,
        max_weight: float | None = 5.0,
    ):
        super().__init__()
        self.margin = margin
        self.min_target_diff = min_target_diff
        self.weight_by_target_diff = weight_by_target_diff
        self.max_weight = max_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds:   [B]
            targets: [B]
        Returns:
            scalar ranking loss
        """
        preds = preds.view(-1)
        targets = targets.view(-1)

        pred_diff = preds.unsqueeze(1) - preds.unsqueeze(0)      # [B, B]
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)

        # only keep informative ordered pairs: y_i > y_j
        mask = target_diff > self.min_target_diff

        if not mask.any():
            return preds.new_zeros(())

        pair_pred_diff = pred_diff[mask]
        pair_target_diff = target_diff[mask]

        # logistic ranking loss: log(1 + exp(-(pred_i - pred_j - margin)))
        loss = F.softplus(-(pair_pred_diff - self.margin))

        if self.weight_by_target_diff:
            weights = pair_target_diff.detach()
            if self.max_weight is not None:
                weights = torch.clamp(weights, max=self.max_weight)
            weights = weights / (weights.mean() + 1e-8)
            loss = loss * weights

        return loss.mean()


class HybridLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.3,
        margin: float = 0.0,
        min_target_diff: float = 0.1,
        weight_by_target_diff: bool = True,
        max_weight: float | None = 5.0,
    ):
        super().__init__()
        self.reg_loss = nn.SmoothL1Loss()
        self.rank_loss = PairwiseRankingLoss(
            margin=margin,
            min_target_diff=min_target_diff,
            weight_by_target_diff=weight_by_target_diff,
            max_weight=max_weight,
        )

        self.alpha = alpha
        self.beta = beta

        self.last_reg_loss = None
        self.last_rank_loss = None
        self.last_total_loss = None

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.view(-1)
        targets = targets.view(-1)

        reg = self.reg_loss(preds, targets)
        rank = self.rank_loss(preds, targets)
        total = self.alpha * reg + self.beta * rank

        self.last_reg_loss = reg.detach().item()
        self.last_rank_loss = rank.detach().item()
        self.last_total_loss = total.detach().item()

        return total