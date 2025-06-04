import yaml
from pydantic import BaseModel


class PreprocessConfig(BaseModel):
    data_location: str
    extension: str
    sampling_rate: int
    signal_length: int
    block_size: int
    oneshot: bool
    out_dir: str
    statistics_dir: str

    @classmethod
    def from_config_path(cls, config_path: str) -> "PreprocessConfig":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "preprocess" not in config:
            raise ValueError("preprocess section not found in config")
        return cls(**config["preprocess"])
