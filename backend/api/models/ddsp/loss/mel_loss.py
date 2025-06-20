import torch
import torch.nn as nn

from api.models.ddsp.core import multiscale_fft, safe_log


class MelLoss(nn.Module):
    def __init__(self, scales: list[int], overlap: float, device: torch.device):
        super().__init__()
        self.scales = scales
        self.overlap = overlap
        self.device = device

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        original_spectrogram: list[torch.Tensor] = multiscale_fft(
            y_pred,
            self.scales,
            self.overlap,
        )
        output_spectrogram: list[torch.Tensor] = multiscale_fft(
            y_target,
            self.scales,
            self.overlap,
        )
        loss: torch.Tensor = torch.zeros(1, device=self.device)
        for s_x, s_y in zip(original_spectrogram, output_spectrogram):
            lin_loss: torch.Tensor = (s_x - s_y).abs().mean()
            log_loss: torch.Tensor = (safe_log(s_x) - safe_log(s_y)).abs().mean()
            loss = loss + lin_loss + log_loss
        return loss
