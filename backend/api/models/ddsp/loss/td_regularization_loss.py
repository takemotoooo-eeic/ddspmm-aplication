import torch
import torch.nn as nn
from api.config.model.loss_config import TargetType
import json
import os

class TimeDomainRegularizationLoss(nn.Module):
    def __init__(
        self,
        device: torch.device,
        target_type: TargetType,
        instrument_name: str,
    ):
        super().__init__()
        self.device = device
        self.target_type = target_type
        self.instrument_name = instrument_name
        # JSONファイルのパス
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        delta_var_path = os.path.join(base_dir, "statistics", "td_delta_var.json")
        delta_delta_var_path = os.path.join(base_dir, "statistics", "td_delta_delta_var.json")
        # JSONファイルの読み込み
        with open(delta_var_path, "r") as f:
            delta_var_data = json.load(f)
        with open(delta_delta_var_path, "r") as f:
            delta_delta_var_data = json.load(f)
        if target_type == TargetType.LOUDNESS:
            self.real_delta_y_var = torch.tensor(delta_var_data["loudness"][instrument_name], device=device)
            self.real_delta_delta_y_var = torch.tensor(delta_delta_var_data["loudness"][instrument_name], device=device)
        elif target_type == TargetType.PITCH:
            self.real_delta_y_var = torch.tensor(delta_var_data["pitch"][instrument_name], device=device)
            self.real_delta_delta_y_var = torch.tensor(delta_delta_var_data["pitch"][instrument_name], device=device)
        elif target_type == TargetType.Z_FEATURE:
            self.real_delta_y_var = torch.tensor(delta_var_data["z_feature"][instrument_name], device=device)
            self.real_delta_delta_y_var = torch.tensor(delta_delta_var_data["z_feature"][instrument_name], device=device)

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
