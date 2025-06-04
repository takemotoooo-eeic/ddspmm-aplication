import torch
import torch.nn as nn


class TimeDomainRegularizationLoss(nn.Module):
    def __init__(
        self,
        device: torch.device,
        real_delta_y_var: torch.Tensor,
        real_delta_delta_y_var: torch.Tensor,
    ):
        super().__init__()
        self.device = device
        self.real_delta_y_var = real_delta_y_var
        self.real_delta_delta_y_var = real_delta_delta_y_var

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        delta_y = torch.diff(y, dim=1)
        delta_delta_y = torch.diff(delta_y, dim=1)

        delta_y_var = torch.var(delta_y, dim=1)
        delta_delta_y_var = torch.var(delta_delta_y, dim=1)

        loss = torch.sum(
            (delta_y_var - self.real_delta_y_var) ** 2
            + (delta_delta_y_var - self.real_delta_delta_y_var) ** 2
        )
        return loss
