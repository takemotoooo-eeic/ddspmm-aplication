import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    hidden_size: int
    n_harmonic: int
    n_bands: int
    sampling_rate: int
    block_size: int

    @classmethod
    def from_config_path(cls, config_path: str) -> "ModelConfig":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "model" not in config:
            raise ValueError("model section not found in config")
        return cls(**config["model"])
