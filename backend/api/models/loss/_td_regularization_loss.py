import torch
import torch.nn as nn
from api.config.model.loss import TargetType
import os
import json
from api.libs.instrument import Instrument


class TimeDomainRegularizationLoss(nn.Module):
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
        delta_var_path = os.path.join(statistics_dir, "td_delta_var.json")
        delta_delta_var_path = os.path.join(statistics_dir, "td_delta_delta_var.json")
        # JSONファイルの読み込み
        with open(delta_var_path, "r") as f:
            delta_var_data = json.load(f)
        with open(delta_delta_var_path, "r") as f:
            delta_delta_var_data = json.load(f)
        
        # 全楽器の統計情報を保持（辞書形式）
        self.real_delta_y_var_dict: dict[str, torch.Tensor] = {}
        self.real_delta_delta_y_var_dict: dict[str, torch.Tensor] = {}
        
        # キー名を決定
        key = "loudness" if target_type == TargetType.LOUDNESS else ("pitch" if target_type == TargetType.PITCH else "z_feature")
        
        for instrument_name in instrument_names:
            # instrument_nameがInstrument enumの場合は.valueを使用、文字列の場合はそのまま使用
            instrument_key = instrument_name.value if isinstance(instrument_name, Instrument) else str(instrument_name)
            
            if key in delta_var_data and instrument_key in delta_var_data[key]:
                self.real_delta_y_var_dict[instrument_key] = torch.tensor(
                    delta_var_data[key][instrument_key], device=device
                )
                self.real_delta_delta_y_var_dict[instrument_key] = torch.tensor(
                    delta_delta_var_data[key][instrument_key], device=device
                )

    def forward(self, y: torch.Tensor, instrument_names: list[int | Instrument]) -> torch.Tensor:
        """
        Args:
            y: (B, L, ...) バッチ内の全サンプルの信号
            instrument_names: バッチ内の各サンプルに対応するinstrument_nameのリスト
        """
        if y.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        # 各サンプルごとに処理
        for i in range(y.shape[0]):
            instrument_name = instrument_names[i].value if isinstance(instrument_names[i], Instrument) else str(instrument_names[i])
            y_sample = y[i:i+1]  # (1, L, ...)
            
            if instrument_name not in self.real_delta_y_var_dict:
                continue
            
            delta_y = torch.diff(y_sample.squeeze(0), dim=0)  # (L-1, ...)
            delta_delta_y = torch.diff(delta_y, dim=0)  # (L-2, ...)
            
            delta_y_var = torch.var(delta_y, dim=0)
            delta_delta_y_var = torch.var(delta_delta_y, dim=0)
            
            real_delta_y_var = self.real_delta_y_var_dict[instrument_name]
            real_delta_delta_y_var = self.real_delta_delta_y_var_dict[instrument_name]
            
            # 形状を揃える
            # loudness/pitchの場合はスカラー、z_featureの場合はベクトルとして保存されている
            # 計算結果の形状を統計情報の形状に合わせる
            if real_delta_y_var.dim() == 0:  # スカラーの場合（loudness/pitch）
                # 計算結果が(1,)の場合はスカラーに変換
                if delta_y_var.dim() == 1 and delta_y_var.shape[0] == 1:
                    delta_y_var = delta_y_var.squeeze(0)
                elif delta_y_var.dim() > 0:
                    delta_y_var = delta_y_var.squeeze()
            else:  # ベクトルの場合（z_feature）
                # 形状が一致することを確認
                if delta_y_var.shape != real_delta_y_var.shape:
                    raise ValueError(f"Mismatch in delta_y_var and real_delta_y_var shapes: delta_y_var.shape: {delta_y_var.shape}, real_delta_y_var.shape: {real_delta_y_var.shape}")
            
            if real_delta_delta_y_var.dim() == 0:  # スカラーの場合（loudness/pitch）
                # 計算結果が(1,)の場合はスカラーに変換
                if delta_delta_y_var.dim() == 1 and delta_delta_y_var.shape[0] == 1:
                    delta_delta_y_var = delta_delta_y_var.squeeze(0)
                elif delta_delta_y_var.dim() > 0:
                    delta_delta_y_var = delta_delta_y_var.squeeze()
            else:  # ベクトルの場合（z_feature）
                # 形状が一致することを確認
                if delta_delta_y_var.shape != real_delta_delta_y_var.shape:
                    raise ValueError(f"Mismatch in delta_delta_y_var and real_delta_delta_y_var shapes: delta_delta_y_var.shape: {delta_delta_y_var.shape}, real_delta_delta_y_var.shape: {real_delta_delta_y_var.shape}")
            
            sample_loss = torch.sum(
                (delta_y_var - real_delta_y_var) ** 2
                + (delta_delta_y_var - real_delta_delta_y_var) ** 2
            )
            total_loss = total_loss + sample_loss
        
        return total_loss
