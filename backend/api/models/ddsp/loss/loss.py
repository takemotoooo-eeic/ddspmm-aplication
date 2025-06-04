import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict

from api.config.model.loss_config import (
    FDLossConfig,
    LossConfig,
    MelLossConfig,
    TargetType,
    TDLossConfig,
)

from .fd_regularization_loss import FrequencyDomainRegularizationLoss
from .mel_loss import MelLoss
from .td_regularization_loss import TimeDomainRegularizationLoss


class MelLossInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, copy_on_model_validation=False
    )
    y_pred: torch.Tensor
    y_target: torch.Tensor


class TDLossInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, copy_on_model_validation=False
    )
    y: torch.Tensor


class FDLossInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, copy_on_model_validation=False
    )
    y: torch.Tensor


class LossInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, copy_on_model_validation=False
    )
    inputs: list[MelLossInputs | TDLossInputs | FDLossInputs]

    @classmethod
    def from_results(
        cls,
        loss_config: LossConfig,
        signal_pred: torch.Tensor | None = None,
        signal_target: torch.Tensor | None = None,
        loudness: torch.Tensor | None = None,
        pitch: torch.Tensor | None = None,
        z_feature: torch.Tensor | None = None,
    ) -> "LossInputs":
        inputs: list[MelLossInputs | TDLossInputs | FDLossInputs] = []
        for loss_item in loss_config.loss:
            if isinstance(loss_item, MelLossConfig):
                inputs.append(MelLossInputs(y_pred=signal_pred, y_target=signal_target))
            elif isinstance(loss_item, TDLossConfig):
                if loss_item.target == TargetType.LOUDNESS:
                    inputs.append(TDLossInputs(y=loudness))
                elif loss_item.target == TargetType.PITCH:
                    inputs.append(TDLossInputs(y=pitch))
                elif loss_item.target == TargetType.Z_FEATURE:
                    inputs.append(TDLossInputs(y=z_feature))
            elif isinstance(loss_item, FDLossConfig):
                if loss_item.target == TargetType.LOUDNESS:
                    inputs.append(FDLossInputs(y=loudness))
                elif loss_item.target == TargetType.PITCH:
                    inputs.append(FDLossInputs(y=pitch))
                elif loss_item.target == TargetType.Z_FEATURE:
                    inputs.append(FDLossInputs(y=z_feature))
        return cls(inputs=inputs)


class Loss(nn.Module):
    def __init__(self, device: torch.device, loss_config: LossConfig):
        super().__init__()
        self.device = device
        self.loss_components: list[
            MelLoss | TimeDomainRegularizationLoss | FrequencyDomainRegularizationLoss
        ] = []
        self.loss_configs: list[MelLossConfig | TDLossConfig | FDLossConfig] = []

        for loss_item in loss_config.loss:
            if isinstance(loss_item, MelLossConfig):
                self.loss_components.append(
                    MelLoss(
                        device=device,
                        scales=loss_item.scales,
                        overlap=loss_item.overlap,
                    )
                )
            elif isinstance(loss_item, TDLossConfig):
                self.loss_components.append(
                    TimeDomainRegularizationLoss(
                        device=device, target_type=loss_item.target
                    )
                )
            elif isinstance(loss_item, FDLossConfig):
                self.loss_components.append(
                    FrequencyDomainRegularizationLoss(
                        device=device, target_type=loss_item.target
                    )
                )
            self.loss_configs.append(loss_item)

    def forward(self, inputs: LossInputs) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for loss_component, loss_config, input in zip(
            self.loss_components, self.loss_configs, inputs.inputs
        ):
            if isinstance(loss_config, MelLossConfig):
                component_loss = loss_component(
                    y_pred=input.y_pred, y_target=input.y_target
                )
            elif isinstance(loss_config, TDLossConfig):
                component_loss = loss_component(
                    y=input.y,
                )
            elif isinstance(loss_config, FDLossConfig):
                component_loss = loss_component(
                    y=input.y,
                )

            total_loss = total_loss + loss_config.ratio * component_loss

        return total_loss
