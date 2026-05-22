import torch
import torch.nn as nn
import torch.nn.functional as F


class L2NormalizationLoss(nn.Module):
    """L2正則化ロス（予測信号のL2ノルム）"""

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: 予測値テンソル
        Returns:
            L2正則化ロス（予測値のL2ノルムの平均）
        """
        # L2ノルムの二乗を計算（全要素の二乗和）
        l2_norm_squared = torch.sum(y_pred ** 2)
        # 要素数で正規化（平均的なL2ノルムの二乗）
        num_elements = y_pred.numel()
        return l2_norm_squared / num_elements
