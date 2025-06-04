from api.controllers.common import CustomBaseModel


class Feature(CustomBaseModel):
    pitch: list[float]
    loudness: list[float]
    z_feature: list[list[float]]


class Features(CustomBaseModel):
    features: list[Feature]


class DDSPGenerateParams(CustomBaseModel):
    pitch: list[float]
    loudness: list[float]
    z_feature: list[list[float]]
