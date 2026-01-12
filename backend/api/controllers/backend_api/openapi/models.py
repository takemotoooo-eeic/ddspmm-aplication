from api.controllers.common import CustomBaseModel
from typing import Generator


class Note(CustomBaseModel):
    start: float
    frequency: float
    duration: float


class Feature(CustomBaseModel):
    instrument_name: str
    pitch: list[float]
    loudness: list[float]
    z_feature: list[list[float]]
    notes: list[Note]


class Features(CustomBaseModel):
    features: list[Feature]


class DDSPGenerateParams(CustomBaseModel):
    pitch: list[float]
    loudness: list[float]
    z_feature: list[list[float]]


class TrainDDSPParams(CustomBaseModel):
    epochs: int
    lr: float


class TrainingProgress(CustomBaseModel):
    current_epoch: int
    total_epochs: int
    loss: float


TrainDDSPOutputStream = Generator[TrainingProgress | Features, None, None]
