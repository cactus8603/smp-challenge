# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class PairwiseRankingLoss(nn.Module):
#     def __init__(
#         self,
#         margin: float = 0.0,
#         min_target_diff: float = 0.0,
#         weight_by_target_diff: bool = True,
#         max_weight: float | None = 5.0,
#     ):
#         super().__init__()
#         self.margin = margin
#         self.min_target_diff = min_target_diff
#         self.weight_by_target_diff = weight_by_target_diff
#         self.max_weight = max_weight

#     def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         preds = preds.view(-1)
#         targets = targets.view(-1)

#         pred_diff = preds.unsqueeze(1) - preds.unsqueeze(0)      # [B, B]
#         target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)

#         mask = target_diff > self.min_target_diff

#         if not mask.any():
#             return preds.new_zeros(())

#         pair_pred_diff = pred_diff[mask]
#         pair_target_diff = target_diff[mask]

#         loss = F.softplus(-(pair_pred_diff - self.margin))

#         if self.weight_by_target_diff:
#             weights = pair_target_diff.detach()
#             if self.max_weight is not None:
#                 weights = torch.clamp(weights, max=self.max_weight)
#             weights = weights / (weights.mean() + 1e-8)
#             loss = loss * weights

#         return loss.mean()


# class WeightedSmoothL1Loss(nn.Module):
#     def __init__(
#         self,
#         high_target_scale: float = 0.5,
#         max_weight: float = 3.0,
#     ):
#         super().__init__()
#         self.high_target_scale = high_target_scale
#         self.max_weight = max_weight

#     def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         preds = preds.view(-1)
#         targets = targets.view(-1)

#         base = F.smooth_l1_loss(preds, targets, reduction="none")

#         # emphasize higher-target samples
#         centered = targets - targets.mean()
#         weights = 1.0 + self.high_target_scale * torch.relu(centered)
#         weights = torch.clamp(weights, max=self.max_weight)
#         weights = weights / (weights.mean() + 1e-8)

#         return (base * weights).mean()


# class VarianceAlignmentLoss(nn.Module):
#     def __init__(self, eps: float = 1e-8):
#         super().__init__()
#         self.eps = eps

#     def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         preds = preds.view(-1)
#         targets = targets.view(-1)

#         pred_std = preds.std(unbiased=False)
#         target_std = targets.std(unbiased=False)

#         return torch.abs(pred_std - target_std + self.eps) - self.eps


# class HybridLoss(nn.Module):
#     def __init__(
#         self,
#         alpha: float = 1.0,     # regression
#         beta: float = 0.3,      # ranking
#         gamma: float = 0.1,     # variance alignment
#         margin: float = 0.0,
#         min_target_diff: float = 0.1,
#         weight_by_target_diff: bool = True,
#         max_weight: float | None = 5.0,
#         high_target_scale: float = 0.5,
#         reg_max_weight: float = 3.0,
#     ):
#         super().__init__()

#         self.reg_loss = WeightedSmoothL1Loss(
#             high_target_scale=high_target_scale,
#             max_weight=reg_max_weight,
#         )
#         self.rank_loss = PairwiseRankingLoss(
#             margin=margin,
#             min_target_diff=min_target_diff,
#             weight_by_target_diff=weight_by_target_diff,
#             max_weight=max_weight,
#         )
#         self.var_loss = VarianceAlignmentLoss()

#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma

#         self.last_reg_loss = None
#         self.last_rank_loss = None
#         self.last_var_loss = None
#         self.last_total_loss = None

#     def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         preds = preds.view(-1)
#         targets = targets.view(-1)

#         reg = self.reg_loss(preds, targets)
#         rank = self.rank_loss(preds, targets)
#         var = self.var_loss(preds, targets)

#         total = self.alpha * reg + self.beta * rank + self.gamma * var

#         self.last_reg_loss = reg.detach().item()
#         self.last_rank_loss = rank.detach().item()
#         self.last_var_loss = var.detach().item()
#         self.last_total_loss = total.detach().item()

#         return total

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardRegressionLoss(nn.Module):
    """
    Regression loss with dynamic hard-sample weighting.

    Goals:
    1. penalize samples with large residuals more
    2. optionally emphasize high-target samples a bit
    3. discourage collapse to predicting only the mean
    """
    def __init__(
        self,
        beta: float = 1.0,
        hard_scale: float = 1.5,
        high_target_scale: float = 0.3,
        max_weight: float = 6.0,
        detach_weight: bool = True,
    ):
        super().__init__()
        self.beta = beta
        self.hard_scale = hard_scale
        self.high_target_scale = high_target_scale
        self.max_weight = max_weight
        self.detach_weight = detach_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.view(-1)
        targets = targets.view(-1)

        base = F.smooth_l1_loss(preds, targets, reduction="none", beta=self.beta)

        residual = torch.abs(preds - targets)
        if self.detach_weight:
            residual = residual.detach()

        centered_targets = targets - targets.mean()

        weights = (
            1.0
            + self.hard_scale * residual
            + self.high_target_scale * torch.relu(centered_targets)
        )

        weights = torch.clamp(weights, max=self.max_weight)
        weights = weights / (weights.mean() + 1e-8)

        return (base * weights).mean()


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss for better Spearman correlation.

    Focuses on pairs with meaningful target differences.
    Larger target gaps can receive larger weights.
    """
    def __init__(
        self,
        margin: float = 0.0,
        min_target_diff: float = 0.2,
        weight_by_target_diff: bool = True,
        max_weight: float | None = 6.0,
    ):
        super().__init__()
        self.margin = margin
        self.min_target_diff = min_target_diff
        self.weight_by_target_diff = weight_by_target_diff
        self.max_weight = max_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.view(-1)
        targets = targets.view(-1)

        pred_diff = preds.unsqueeze(1) - preds.unsqueeze(0)
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)

        mask = target_diff > self.min_target_diff

        if not mask.any():
            return preds.new_zeros(())

        pair_pred_diff = pred_diff[mask]
        pair_target_diff = target_diff[mask]

        # if pair_pred_diff is too small or wrong sign, punish strongly
        loss = F.softplus(-(pair_pred_diff - self.margin))

        if self.weight_by_target_diff:
            weights = pair_target_diff.detach()
            if self.max_weight is not None:
                weights = torch.clamp(weights, max=self.max_weight)
            weights = weights / (weights.mean() + 1e-8)
            loss = loss * weights

        return loss.mean()


class BatchContrastLoss(nn.Module):
    """
    Prevent prediction collapse by forcing batch-level spread.

    If preds are too concentrated around their mean, this loss increases.
    It compares pairwise distance structure between preds and targets.
    """
    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.view(-1)
        targets = targets.view(-1)

        pred_dist = torch.abs(preds.unsqueeze(1) - preds.unsqueeze(0))
        target_dist = torch.abs(targets.unsqueeze(1) - targets.unsqueeze(0))

        pred_dist = pred_dist / (pred_dist.mean() + 1e-8)
        target_dist = target_dist / (target_dist.mean() + 1e-8)

        return F.smooth_l1_loss(pred_dist, target_dist)


class VarianceFloorLoss(nn.Module):
    """
    Stronger anti-collapse loss.

    If prediction std is much smaller than target std,
    punish aggressively. If it's already large enough,
    give zero penalty.
    """
    def __init__(self, ratio: float = 0.7):
        super().__init__()
        self.ratio = ratio

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.view(-1)
        targets = targets.view(-1)

        pred_std = preds.std(unbiased=False)
        target_std = targets.std(unbiased=False).detach()

        desired_std = self.ratio * target_std
        return F.relu(desired_std - pred_std)


class MeanEscapeLoss(nn.Module):
    """
    Extra penalty when predictions stay too close to batch mean
    while targets are much more spread out.

    This directly attacks 'everything predicted near the mean'.
    """
    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.view(-1)
        targets = targets.view(-1)

        pred_centered = torch.abs(preds - preds.mean())
        target_centered = torch.abs(targets - targets.mean()).detach()

        pred_centered = pred_centered / (pred_centered.mean() + 1e-8)
        target_centered = target_centered / (target_centered.mean() + 1e-8)

        return F.smooth_l1_loss(pred_centered, target_centered)


class LargeErrorFocalLoss(nn.Module):
    """
    Additional focal-style term:
    samples with larger errors get disproportionately larger penalties.
    """
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.view(-1)
        targets = targets.view(-1)

        abs_err = torch.abs(preds - targets)
        base = F.smooth_l1_loss(preds, targets, reduction="none")

        scale = (abs_err / (abs_err.mean() + 1e-8)).detach()
        focal_weight = torch.pow(scale + 1e-6, self.gamma)
        focal_weight = focal_weight / (focal_weight.mean() + 1e-8)

        return (base * focal_weight).mean()


class HybridLoss(nn.Module):
    """
    Main loss:
    - hard regression
    - pairwise ranking
    - batch contrast (spread structure)
    - variance floor (anti-collapse)
    - mean escape (anti-mean-prediction)
    - focal large-error emphasis
    """
    def __init__(
        self,
        alpha: float = 1.0,   # hard regression
        beta: float = 0.6,    # ranking
        gamma: float = 0.25,  # batch contrast
        delta: float = 0.20,  # variance floor
        eta: float = 0.20,    # mean escape
        zeta: float = 0.35,   # focal large-error term

        reg_beta: float = 1.0,
        hard_scale: float = 1.5,
        high_target_scale: float = 0.3,
        reg_max_weight: float = 6.0,

        rank_margin: float = 0.0,
        min_target_diff: float = 0.2,
        weight_by_target_diff: bool = True,
        rank_max_weight: float | None = 6.0,

        variance_floor_ratio: float = 0.7,
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        self.reg_loss = HardRegressionLoss(
            beta=reg_beta,
            hard_scale=hard_scale,
            high_target_scale=high_target_scale,
            max_weight=reg_max_weight,
            detach_weight=True,
        )

        self.rank_loss = PairwiseRankingLoss(
            margin=rank_margin,
            min_target_diff=min_target_diff,
            weight_by_target_diff=weight_by_target_diff,
            max_weight=rank_max_weight,
        )

        self.batch_contrast_loss = BatchContrastLoss()
        self.variance_floor_loss = VarianceFloorLoss(ratio=variance_floor_ratio)
        self.mean_escape_loss = MeanEscapeLoss()
        self.large_error_focal_loss = LargeErrorFocalLoss(gamma=focal_gamma)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta
        self.zeta = zeta

        self.last_reg_loss = None
        self.last_rank_loss = None
        self.last_batch_contrast_loss = None
        self.last_variance_floor_loss = None
        self.last_mean_escape_loss = None
        self.last_large_error_focal_loss = None
        self.last_total_loss = None

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.view(-1)
        targets = targets.view(-1)

        reg = self.reg_loss(preds, targets)
        rank = self.rank_loss(preds, targets)
        batch_contrast = self.batch_contrast_loss(preds, targets)
        variance_floor = self.variance_floor_loss(preds, targets)
        mean_escape = self.mean_escape_loss(preds, targets)
        focal = self.large_error_focal_loss(preds, targets)

        total = (
            self.alpha * reg
            + self.beta * rank
            + self.gamma * batch_contrast
            + self.delta * variance_floor
            + self.eta * mean_escape
            + self.zeta * focal
        )

        self.last_reg_loss = reg.detach().item()
        self.last_rank_loss = rank.detach().item()
        self.last_batch_contrast_loss = batch_contrast.detach().item()
        self.last_variance_floor_loss = variance_floor.detach().item()
        self.last_mean_escape_loss = mean_escape.detach().item()
        self.last_large_error_focal_loss = focal.detach().item()
        self.last_total_loss = total.detach().item()

        return total