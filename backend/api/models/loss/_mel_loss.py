from api.libs.core import multiscale_fft, multiscale_fft_v2, safe_log
import torch
import torch.nn as nn


def mel_loss(
    signal: torch.Tensor,
    y: torch.Tensor,
    scales: list[int],
    overlap: float,
    device: torch.device,
) -> torch.Tensor:
    original_spectrogram: list[torch.Tensor] = multiscale_fft(
        signal,
        scales,
        overlap,
    )
    output_spectrogram: list[torch.Tensor] = multiscale_fft(
        y,
        scales,
        overlap,
    )
    loss: torch.Tensor = torch.zeros(1, device=device)
    for s_x, s_y in zip(original_spectrogram, output_spectrogram):
        lin_loss: torch.Tensor = (s_x - s_y).abs().mean()
        log_loss: torch.Tensor = (safe_log(s_x) - safe_log(s_y)).abs().mean()
        loss = loss + lin_loss + log_loss
    return loss


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


class MelLossV2(nn.Module):
    def __init__(self, scales: list[int], overlap: float, device: torch.device):
        super().__init__()
        self.scales = scales
        self.overlap = overlap
        self.device = device

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        original_spectrogram: list[torch.Tensor] = multiscale_fft_v2(
            y_pred,
            self.scales,
            self.overlap,
        )
        output_spectrogram: list[torch.Tensor] = multiscale_fft_v2(
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

