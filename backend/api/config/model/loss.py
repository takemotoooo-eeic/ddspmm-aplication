from enum import Enum
from pydantic import BaseModel, field_validator, StrictFloat, StrictInt
import yaml


class LossType(str, Enum):
    MEL = "mel"
    MEL_V2 = "mel_v2"
    TD = "td"
    FD = "fd"
    MSE = "mse"
    L2 = "l2"
    DELTA = "delta"


class TargetType(str, Enum):
    LOUDNESS = "loudness"
    PITCH = "pitch"
    Z_FEATURE = "z_feature"


class MelLossConfig(BaseModel):
    type: LossType = LossType.MEL
    ratio: StrictFloat
    scales: list[StrictInt]
    overlap: StrictFloat


class MelLossV2Config(BaseModel):
    type: LossType = LossType.MEL_V2
    ratio: StrictFloat
    scales: list[StrictInt]
    overlap: StrictFloat


class TDLossConfig(BaseModel):
    type: LossType = LossType.TD
    ratio: StrictFloat
    target: TargetType


class FDLossConfig(BaseModel):
    type: LossType = LossType.FD
    ratio: StrictFloat
    target: TargetType


class MSELossConfig(BaseModel):
    type: LossType = LossType.MSE
    ratio: StrictFloat
    target: TargetType


class L2NormalizationLossConfig(BaseModel):
    type: LossType = LossType.L2
    ratio: StrictFloat
    target: TargetType

class DeltaLossConfig(BaseModel):
    type: LossType = LossType.DELTA
    ratio: StrictFloat
    target: TargetType


class LossConfig(BaseModel):
    loss: list[MelLossConfig | MelLossV2Config | TDLossConfig | FDLossConfig | MSELossConfig | L2NormalizationLossConfig | DeltaLossConfig]

    @field_validator("loss", mode="before")
    @classmethod
    def validate_loss_configs(cls, v):
        if isinstance(v, list):
            validated_losses = []
            for item in v:
                if isinstance(item, dict):
                    loss_type = item.get("type")
                            
                    if loss_type == "mel":
                        validated_losses.append(MelLossConfig(**item))
                    elif loss_type == "mel_v2":
                        validated_losses.append(MelLossV2Config(**item))
                    elif loss_type == "td":
                        validated_losses.append(TDLossConfig(**item))
                    elif loss_type == "fd":
                        validated_losses.append(FDLossConfig(**item))
                    elif loss_type == "mse":
                        validated_losses.append(MSELossConfig(**item))
                    elif loss_type == "l2":
                        validated_losses.append(L2NormalizationLossConfig(**item))
                    elif loss_type == "delta":
                        validated_losses.append(DeltaLossConfig(**item))
                    else:
                        raise ValueError(f"Unknown loss type: {loss_type}")
                else:
                    validated_losses.append(item)
            return validated_losses
        return v

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        for loss_item in data["loss"]:
            if "type" in loss_item:
                loss_item["type"] = loss_item["type"].value
            if "target" in loss_item:
                loss_item["target"] = loss_item["target"].value
        return data

    @classmethod
    def from_config_path(cls, config_path: str) -> "LossConfig":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "loss" not in config:
            raise ValueError("loss section not found in config")
        loss_section = config["loss"]
        # Hydra 形式: loss: { _target_: ..., loss: [ ... ] }
        if isinstance(loss_section, dict):
            if "loss" in loss_section:
                loss_list = loss_section["loss"]
            else:
                raise ValueError(
                    "loss section must contain a 'loss' list "
                    "(Hydra-style nested config)"
                )
        elif isinstance(loss_section, list):
            loss_list = loss_section
        else:
            raise ValueError(f"Unexpected loss section type: {type(loss_section)}")
        return cls(loss=loss_list)

