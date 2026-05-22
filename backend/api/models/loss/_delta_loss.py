import torch
import torch.nn as nn

class DeltaLoss(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        delta_y_pred = torch.diff(y_pred, dim=1)
        delta_y_target = torch.diff(y_target, dim=1)

        delta_delta_y_pred = torch.diff(delta_y_pred, dim=1)
        delta_delta_y_target = torch.diff(delta_y_target, dim=1)

        loss = nn.MSELoss()(delta_y_pred, delta_y_target) + nn.MSELoss()(delta_delta_y_pred, delta_delta_y_target)
        return loss
