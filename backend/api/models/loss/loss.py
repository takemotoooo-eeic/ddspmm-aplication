import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from api.config.model.loss import (
    LossConfig,
    LossType,
    MelLossConfig,
    MelLossV2Config,
    MSELossConfig,
    L2NormalizationLossConfig,
    TDLossConfig,
    FDLossConfig,
    TargetType,
    DeltaLossConfig,
)
from ._mel_loss import MelLoss, MelLossV2
from ._mse_loss import MSELoss
from ._l2_normalization_loss import L2NormalizationLoss
from ._td_regularization_loss import TimeDomainRegularizationLoss
from ._fd_regularization_loss import FrequencyDomainRegularizationLoss
from ._delta_loss import DeltaLoss
from api.libs.instrument import Instrument


class MelLossInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, copy_on_model_validation=False
    )
    y_pred: torch.Tensor
    y_target: torch.Tensor


class MelLossV2Inputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, copy_on_model_validation=False
    )
    y_pred: torch.Tensor
    y_target: torch.Tensor


class MSELossInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, copy_on_model_validation=False
    )
    y_pred: torch.Tensor
    y_target: torch.Tensor


class L2NormalizationLossInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, copy_on_model_validation=False
    )
    y_pred: torch.Tensor


class TDLossInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, copy_on_model_validation=False
    )
    y: torch.Tensor  # (B, L, ...) バッチ内の全サンプル
    instrument_names: list[int | str]  # バッチ内の各サンプルに対応するinstrument_idのリスト


class FDLossInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, copy_on_model_validation=False
    )
    y: torch.Tensor  # (B, L, ...) バッチ内の全サンプル
    instrument_names: list[int | str]  # バッチ内の各サンプルに対応するinstrument_idのリスト


class DeltaFeatureLossInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, copy_on_model_validation=False
    )
    y_pred: torch.Tensor
    y_target: torch.Tensor

class LossInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, copy_on_model_validation=False
    )
    inputs: list[
        MelLossInputs
        | MelLossV2Inputs
        | MSELossInputs
        | L2NormalizationLossInputs
        | TDLossInputs
        | FDLossInputs
        | DeltaFeatureLossInputs
    ]

    @classmethod
    def from_results(
        cls,
        loss_config: LossConfig,
        signal_pred: torch.Tensor | None = None,
        signal_target: torch.Tensor | None = None,
        # TDLoss/FDLoss用（targetのみ）
        loudness: torch.Tensor | list[torch.Tensor] | None = None,
        pitch: torch.Tensor | list[torch.Tensor] | None = None,
        z_feature: torch.Tensor | list[torch.Tensor] | None = None,
        # MSELoss/L2NormalizationLoss用（pred/targetペア）
        pitch_pred: torch.Tensor | None = None,
        pitch_target: torch.Tensor | None = None,
        loudness_pred: torch.Tensor | None = None,
        loudness_target: torch.Tensor | None = None,
        z_feature_pred: torch.Tensor | None = None,
        z_feature_target: torch.Tensor | None = None,
        instrument_names: list[int | Instrument] | None = None,
    ) -> "LossInputs":
        inputs: list[
            MelLossInputs
            | MelLossV2Inputs
            | MSELossInputs
            | L2NormalizationLossInputs
            | TDLossInputs
            | FDLossInputs
        ] = []
        for loss_item in loss_config.loss:
            if isinstance(loss_item, MelLossConfig):
                if signal_pred is not None and signal_target is not None:
                    inputs.append(
                        MelLossInputs(y_pred=signal_pred, y_target=signal_target)
                    )
            elif isinstance(loss_item, MelLossV2Config):
                if signal_pred is not None and signal_target is not None:
                    inputs.append(
                        MelLossV2Inputs(y_pred=signal_pred, y_target=signal_target)
                    )
            elif isinstance(loss_item, MSELossConfig):
                # targetが指定されている場合は、対応するpred/targetペアを使用
                if loss_item.target is None:
                    # targetが指定されていない場合はsignal_pred/signal_targetを使用
                    if signal_pred is None or signal_target is None:
                        raise ValueError("signal_pred and signal_target are required for MSELossConfig")
                    inputs.append(MSELossInputs(y_pred=signal_pred, y_target=signal_target))
                elif loss_item.target == TargetType.LOUDNESS:
                    if loudness_pred is None or loudness_target is None:
                        raise ValueError("loudness_pred and loudness_target are required for MSELossConfig")
                    inputs.append(MSELossInputs(y_pred=loudness_pred, y_target=loudness_target))
                elif loss_item.target == TargetType.PITCH:
                    if pitch_pred is None or pitch_target is None:
                        raise ValueError("pitch_pred and pitch_target are required for MSELossConfig")
                    inputs.append(MSELossInputs(y_pred=pitch_pred, y_target=pitch_target))
                elif loss_item.target == TargetType.Z_FEATURE:
                    if z_feature_pred is None or z_feature_target is None:
                        raise ValueError("z_feature_pred and z_feature_target are required for MSELossConfig")
                    inputs.append(MSELossInputs(y_pred=z_feature_pred, y_target=z_feature_target))
            elif isinstance(loss_item, L2NormalizationLossConfig):
                # targetが指定されている場合は、対応するpredを使用
                if loss_item.target is None:
                    raise ValueError("target is required for L2NormalizationLossConfig")
                if loss_item.target == TargetType.LOUDNESS:
                    if loudness_pred is None:
                        raise ValueError("loudness_pred is required for L2NormalizationLossConfig")
                    inputs.append(L2NormalizationLossInputs(y_pred=loudness_pred))
                elif loss_item.target == TargetType.PITCH:
                    if pitch_pred is None:
                        raise ValueError("pitch_pred is required for L2NormalizationLossConfig")
                    inputs.append(L2NormalizationLossInputs(y_pred=pitch_pred))
                elif loss_item.target == TargetType.Z_FEATURE:
                    if z_feature_pred is None:
                        raise ValueError("z_feature_pred is required for L2NormalizationLossConfig")
                    inputs.append(L2NormalizationLossInputs(y_pred=z_feature_pred))

            elif isinstance(loss_item, DeltaLossConfig):
                if loss_item.target is None:
                    raise ValueError("target is required for DeltaLossConfig")
                if loss_item.target == TargetType.LOUDNESS:
                    if loudness_pred is None or loudness_target is None:
                        raise ValueError("loudness_pred and loudness_target are required for DeltaLossConfig")
                    inputs.append(DeltaFeatureLossInputs(y_pred=loudness_pred, y_target=loudness_target))
                elif loss_item.target == TargetType.PITCH:
                    if pitch_pred is None or pitch_target is None:
                        raise ValueError("pitch_pred and pitch_target are required for DeltaLossConfig")
                    inputs.append(DeltaFeatureLossInputs(y_pred=pitch_pred, y_target=pitch_target))
                elif loss_item.target == TargetType.Z_FEATURE:
                    if z_feature_pred is None or z_feature_target is None:
                        raise ValueError("z_feature_pred and z_feature_target are required for DeltaLossConfig")
                    inputs.append(DeltaFeatureLossInputs(y_pred=z_feature_pred, y_target=z_feature_target))
            elif isinstance(loss_item, TDLossConfig):
                if instrument_names is None:
                    raise ValueError("instrument_names is required for TDLossConfig")

                if loss_item.target == TargetType.LOUDNESS:
                    # loudnessがlistの場合はスタック、Tensorの場合はそのまま
                    if loudness is not None:
                        if isinstance(loudness, list):
                            y_batch = torch.stack(loudness, dim=0)
                        else:
                            y_batch = loudness
                        inputs.append(TDLossInputs(y=y_batch, instrument_names=instrument_names))
                elif loss_item.target == TargetType.PITCH:
                    if pitch is not None:
                        if isinstance(pitch, list):
                            y_batch = torch.stack(pitch, dim=0)
                        else:
                            y_batch = pitch
                        inputs.append(TDLossInputs(y=y_batch, instrument_names=instrument_names))
                elif loss_item.target == TargetType.Z_FEATURE:
                    if z_feature is not None:
                        if isinstance(z_feature, list):
                            y_batch = torch.stack(z_feature, dim=0)
                        else:
                            y_batch = z_feature
                        inputs.append(TDLossInputs(y=y_batch, instrument_names=instrument_names))
            elif isinstance(loss_item, FDLossConfig):
                if instrument_names is None:
                    raise ValueError("instrument_names is required for FDLossConfig")
                
                if loss_item.target == TargetType.LOUDNESS:
                    if loudness is not None:
                        if isinstance(loudness, list):
                            y_batch = torch.stack(loudness, dim=0)
                        else:
                            y_batch = loudness
                        inputs.append(FDLossInputs(y=y_batch, instrument_names=instrument_names))
                elif loss_item.target == TargetType.PITCH:
                    if pitch is not None:
                        if isinstance(pitch, list):
                            y_batch = torch.stack(pitch, dim=0)
                        else:
                            y_batch = pitch
                        inputs.append(FDLossInputs(y=y_batch, instrument_names=instrument_names))
                elif loss_item.target == TargetType.Z_FEATURE:
                    if z_feature is not None:
                        if isinstance(z_feature, list):
                            y_batch = torch.stack(z_feature, dim=0)
                        else:
                            y_batch = z_feature
                        inputs.append(FDLossInputs(y=y_batch, instrument_names=instrument_names))
        return cls(inputs=inputs)


class Loss(nn.Module):
    def __init__(
        self,
        device: torch.device,
        loss_config: LossConfig,
        instrument_names: list | None = None,
        statistics_dir: str | None = None,
    ):
        super().__init__()
        self.device = device
        self.loss_components: list[
            MelLoss | MelLossV2 | MSELoss | L2NormalizationLoss | DeltaLoss | TimeDomainRegularizationLoss | FrequencyDomainRegularizationLoss
        ] = []
        self.loss_configs: list[
            MelLossConfig | MelLossV2Config | MSELossConfig | L2NormalizationLossConfig | DeltaLossConfig | TDLossConfig | FDLossConfig
        ] = []

        for loss_item in loss_config.loss:
            loss_type: LossType | None = None
            if isinstance(loss_item, MelLossConfig):
                loss_type = LossType.MEL
            elif isinstance(loss_item, MelLossV2Config):
                loss_type = LossType.MEL_V2
            elif isinstance(loss_item, MSELossConfig):
                loss_type = LossType.MSE
            elif isinstance(loss_item, L2NormalizationLossConfig):
                loss_type = LossType.L2
            elif isinstance(loss_item, DeltaLossConfig):
                loss_type = LossType.DELTA
            elif isinstance(loss_item, TDLossConfig):
                loss_type = LossType.TD
            elif isinstance(loss_item, FDLossConfig):
                loss_type = LossType.FD
            
            if loss_type == LossType.MEL:
                self.loss_components.append(
                    MelLoss(
                        device=device,
                        scales=loss_item.scales,
                        overlap=loss_item.overlap,
                    )
                )
                self.loss_configs.append(loss_item)
            elif loss_type == LossType.MEL_V2:
                self.loss_components.append(
                    MelLossV2(
                        device=device,
                        scales=loss_item.scales,
                        overlap=loss_item.overlap,
                    )
                )
                self.loss_configs.append(loss_item)
            elif loss_type == LossType.MSE:
                self.loss_components.append(MSELoss(device=device))
                self.loss_configs.append(loss_item)
            elif loss_type == LossType.L2:
                self.loss_components.append(L2NormalizationLoss(device=device))
                self.loss_configs.append(loss_item)
            elif loss_type == LossType.DELTA:
                self.loss_components.append(DeltaLoss(device=device))
                self.loss_configs.append(loss_item)
            elif loss_type == LossType.TD:
                if instrument_names is None or statistics_dir is None:
                    raise ValueError("instrument_names and statistics_dir are required for TDLossConfig")
                self.loss_components.append(
                    TimeDomainRegularizationLoss(
                        device=device,
                        target_type=loss_item.target,
                        instrument_names=instrument_names,
                        statistics_dir=statistics_dir,
                    )
                )
                self.loss_configs.append(loss_item)
            elif loss_type == LossType.FD:
                if instrument_names is None or statistics_dir is None:
                    raise ValueError("instrument_names and statistics_dir are required for FDLossConfig")
                self.loss_components.append(
                    FrequencyDomainRegularizationLoss(
                        device=device,
                        target_type=loss_item.target,
                        instrument_names=instrument_names,
                        statistics_dir=statistics_dir,
                    )
                )
                self.loss_configs.append(loss_item)
            else:
                raise ValueError(f"Unknown loss type: {loss_type} (type: {type(loss_item)})")

    def __repr__(self) -> str:
        loss_repr = ""
        for loss_component, loss_config in zip(self.loss_components, self.loss_configs):
            if isinstance(loss_component, MelLoss):
                loss_repr += f"Mel Loss (scales={loss_config.scales}, overlap={loss_config.overlap})\n"
            elif isinstance(loss_component, MelLossV2):
                loss_repr += f"Mel Loss V2 (scales={loss_config.scales}, overlap={loss_config.overlap})\n"
            elif isinstance(loss_component, MSELoss):
                if isinstance(loss_config, MSELossConfig) and loss_config.target is not None:
                    loss_repr += f"MSE Loss (target={loss_config.target.value})\n"
                else:
                    loss_repr += "MSE Loss\n"
            elif isinstance(loss_component, L2NormalizationLoss):
                if isinstance(loss_config, L2NormalizationLossConfig) and loss_config.target is not None:
                    loss_repr += f"L2 Normalization Loss (target={loss_config.target.value})\n"
                else:
                    loss_repr += "L2 Normalization Loss\n"
            elif isinstance(loss_component, DeltaLoss):
                loss_repr += f"Delta Loss (target={loss_config.target.value})\n"
            elif isinstance(loss_component, TimeDomainRegularizationLoss):
                loss_repr += (
                    f"Time Domain Regularization (target={loss_config.target.value})\n"
                )
            elif isinstance(loss_component, FrequencyDomainRegularizationLoss):
                loss_repr += f"Frequency Domain Regularization (target={loss_config.target.value})\n"
            else:
                raise ValueError(f"Unknown loss type: {type(loss_component)}")
        return loss_repr

    def forward(self, inputs: LossInputs) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for loss_component, loss_config, input in zip(
            self.loss_components, self.loss_configs, inputs.inputs
        ):
            if isinstance(loss_config, MelLossConfig):
                component_loss: torch.Tensor = loss_component(
                    y_pred=input.y_pred, y_target=input.y_target
                )
            elif isinstance(loss_config, MelLossV2Config):
                component_loss: torch.Tensor = loss_component(
                    y_pred=input.y_pred,
                    y_target=input.y_target,
                )
            elif isinstance(loss_config, MSELossConfig):
                component_loss: torch.Tensor = loss_component(
                    y_pred=input.y_pred, y_target=input.y_target
                )
            elif isinstance(loss_config, L2NormalizationLossConfig):
                component_loss: torch.Tensor = loss_component(y_pred=input.y_pred)
            elif isinstance(loss_config, DeltaLossConfig):
                component_loss: torch.Tensor = loss_component(
                    y_pred=input.y_pred, y_target=input.y_target,
                )
            elif isinstance(loss_config, TDLossConfig):
                component_loss: torch.Tensor = loss_component(
                    y=input.y,
                    instrument_names=input.instrument_names,
                )
            elif isinstance(loss_config, FDLossConfig):
                component_loss: torch.Tensor = loss_component(
                    y=input.y,
                    instrument_names=input.instrument_names,
                )
            
            # print(f"Component loss({loss_config.type}): {component_loss.item() * loss_config.ratio:.6f}")

            total_loss = total_loss + loss_config.ratio * component_loss

        return total_loss

    def is_valid_for_pretrain(self) -> bool:
        for loss_component in self.loss_components:
            if isinstance(loss_component, TimeDomainRegularizationLoss) or isinstance(
                loss_component, FrequencyDomainRegularizationLoss
            ):
                return False
        return True
