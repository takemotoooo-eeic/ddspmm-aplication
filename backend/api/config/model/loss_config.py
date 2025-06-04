from enum import Enum

import yaml
from pydantic import BaseModel


class LossType(str, Enum):
    MEL = "mel"
    TD = "td"
    FD = "fd"


class TargetType(str, Enum):
    LOUDNESS = "loudness"
    PITCH = "pitch"
    Z_FEATURE = "z_feature"


class MelLossConfig(BaseModel):
    type: LossType = LossType.MEL
    ratio: float
    scales: list[int]
    overlap: float


class TDLossConfig(BaseModel):
    type: LossType = LossType.TD
    ratio: float
    target: TargetType


class FDLossConfig(BaseModel):
    type: LossType = LossType.FD
    ratio: float
    target: TargetType


class LossConfig(BaseModel):
    loss: list[MelLossConfig | TDLossConfig | FDLossConfig]

    @classmethod
    def from_config_path(cls, config_path: str) -> "LossConfig":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "loss" not in config:
            raise ValueError("loss section not found in config")
        return cls(loss=config["loss"])
