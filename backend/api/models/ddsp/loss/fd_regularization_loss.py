import torch
import torch.nn as nn


class FrequencyDomainRegularizationLoss(nn.Module):
    def __init__(self, device: torch.device, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.device = device
        self.mean = mean
        self.std = std
        self.norm_distribution = torch.distributions.Normal(
            loc=self.mean, scale=self.std
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y_fft: torch.Tensor = torch.fft.fft(y, dim=1)
        y_fft = y_fft.abs()
        if y_fft.ndim == 3:
            y_fft = y_fft[:, 1 : y_fft.shape[-1] // 2, :]
        else:
            y_fft = y_fft[:, 1 : y_fft.shape[-1] // 2]
        log_prob: torch.Tensor = self.norm_distribution.log_prob(y_fft)
        loss = -log_prob.sum()
        return loss
