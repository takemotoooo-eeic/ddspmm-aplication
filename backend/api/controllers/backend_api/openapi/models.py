from typing import Literal

from api.controllers.common import CustomBaseModel


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


class DiffusionGenerateParams(CustomBaseModel):
    notes: list[Note]
    instrument_name: str
    signal_length: int
    num_denoising_steps: int | None = None
    use_ddim: bool | None = None
    note_operation: Literal["add", "delete", "move", "resize"] | None = None
    operation_prev_note: Note | None = None
    operation_note: Note | None = None
    prev_features: DDSPGenerateParams | None = None


class FluidsynthGenerateParams(CustomBaseModel):
    notes: list[Note]
    instrument_name: str
    signal_length: int
