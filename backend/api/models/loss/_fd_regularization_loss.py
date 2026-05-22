import torch
import torch.nn as nn
from api.config.model.loss import TargetType
import json
import os
from api.libs.instrument import Instrument


class FrequencyDomainRegularizationLoss(nn.Module):
    def __init__(
        self,
        device: torch.device,
        target_type: TargetType,
        instrument_names: list[Instrument | int],
        statistics_dir: str,
    ):
        super().__init__()
        self.device = device
        self.target_type = target_type
        self.statistics_dir = statistics_dir
        # JSONファイルのパス
        fd_std_path = os.path.join(statistics_dir, "fd_std.json")
        fd_mean_path = os.path.join(statistics_dir, "fd_mean.json")

        # 全楽器の統計情報を保持（辞書形式）
        self.mean_dict: dict[str, torch.Tensor] = {}
        self.std_dict: dict[str, torch.Tensor] = {}
        
        with open(fd_mean_path, "r") as f:
            fd_mean_data = json.load(f)
        with open(fd_std_path, "r") as f:
            fd_std_data = json.load(f)
        
        # キー名を決定
        key = "loudness" if target_type == TargetType.LOUDNESS else ("pitch" if target_type == TargetType.PITCH else "z_feature")
        
        for instrument_name in instrument_names:
            # instrument_nameがInstrument enumの場合は.valueを使用、文字列の場合はそのまま使用
            instrument_key = instrument_name.value if isinstance(instrument_name, Instrument) else str(instrument_name)
            
            if key in fd_mean_data and instrument_key in fd_mean_data[key]:
                self.mean_dict[instrument_key] = torch.tensor(
                    fd_mean_data[key][instrument_key],
                    device=device,
                    dtype=torch.float32,
                )
                self.std_dict[instrument_key] = torch.tensor(
                    fd_std_data[key][instrument_key],
                    device=device,
                    dtype=torch.float32,
                )
                # ゼロ除算を避けるために、標準偏差に下限値を設ける
                self.std_dict[instrument_key] = torch.clamp(self.std_dict[instrument_key], min=1e-6)

    def forward(self, y: torch.Tensor, instrument_names: list[Instrument | int]) -> torch.Tensor:
        """
        Args:
            y: (B, L, ...) バッチ内の全サンプルの信号
            instrument_names: バッチ内の各サンプルに対応するinstrument_nameのリスト
        """
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        # 各サンプルごとに処理
        for i in range(y.shape[0]):
            instrument_name = instrument_names[i].value if isinstance(instrument_names[i], Instrument) else str(instrument_names[i])
            y_sample = y[i:i+1]  # (1, L, ...)
            
            if instrument_name not in self.mean_dict:
                continue
            
            y_fft: torch.Tensor = torch.fft.fft(y_sample.squeeze(0), dim=0)  # (L, ...)
            y_fft = y_fft.abs()
            if y_fft.ndim == 2:
                y_fft = y_fft[1 : y_fft.shape[0] // 2, :]
            else:
                y_fft = y_fft[1 : y_fft.shape[0] // 2]
            
            mean = self.mean_dict[instrument_name]
            std = self.std_dict[instrument_name]
            
            if y_fft.shape != mean.shape:
                raise ValueError(f"Mismatch in y_fft and mean shapes: y_fft.shape: {y_fft.shape}, mean.shape: {mean.shape}")
            
            diff = y_fft - mean
            loss = (diff**2) / (2 * std**2)
            
            sample_loss = loss.sum()
            total_loss = total_loss + sample_loss
        
        return total_loss
