import torch
import torch.nn as nn
from api.config.model.loss_config import TargetType
import json
import os


class FrequencyDomainRegularizationLoss(nn.Module):
    def __init__(self, device: torch.device, target_type: TargetType, instrument_name: str):
        super().__init__()
        self.device = device
        # JSONファイルのパス
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fd_std_path = os.path.join(base_dir, "statistics", "fd_std.json")
        fd_mean_path = os.path.join(base_dir, "statistics", "fd_mean.json")

        with open(fd_mean_path, "r") as f:
            fd_mean_data = json.load(f)
        if target_type == TargetType.LOUDNESS:
            self.mean = torch.tensor(fd_mean_data["loudness"][instrument_name], device=device, dtype=torch.float32)
        elif target_type == TargetType.PITCH:
            self.mean = torch.tensor(fd_mean_data["pitch"][instrument_name], device=device, dtype=torch.float32)
        elif target_type == TargetType.Z_FEATURE:
            self.mean = torch.tensor(fd_mean_data["z_feature"][instrument_name], device=device, dtype=torch.float32)

        with open(fd_std_path, "r") as f:
            fd_std_data = json.load(f)
        if target_type == TargetType.LOUDNESS:
            self.std = torch.tensor(fd_std_data["loudness"][instrument_name], device=device, dtype=torch.float32)
        elif target_type == TargetType.PITCH:
            self.std = torch.tensor(fd_std_data["pitch"][instrument_name], device=device, dtype=torch.float32)
        elif target_type == TargetType.Z_FEATURE:
            self.std = torch.tensor(fd_std_data["z_feature"][instrument_name], device=device, dtype=torch.float32)
        
        # ゼロ除算を避けるために、標準偏差に下限値を設ける
        self.std = torch.clamp(self.std, min=1e-6)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y_fft: torch.Tensor = torch.fft.fft(y, dim=1)
        y_fft = y_fft.abs()
        if y_fft.ndim == 3:
            y_fft = y_fft[:, 1 : y_fft.shape[1] // 2, :]
        else:
            y_fft = y_fft[:, 1 : y_fft.shape[1] // 2]
        
        diff = y_fft - self.mean
        loss = (diff ** 2) / (2 * self.std ** 2)
        
        loss_val = loss.sum()
        return loss_val
