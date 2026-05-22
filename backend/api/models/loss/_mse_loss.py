import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """Mean Squared Error Loss"""

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: 予測値テンソル
            y_target: ターゲット値テンソル
        """
        return torch.nn.functional.mse_loss(y_pred, y_target)
