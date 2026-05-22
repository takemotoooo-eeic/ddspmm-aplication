from pydantic import BaseModel, StrictInt
import yaml


class ModelConfig(BaseModel):
    hidden_size: StrictInt
    n_harmonic: StrictInt
    n_bands: StrictInt
    sampling_rate: StrictInt
    block_size: StrictInt

    @classmethod
    def from_config_path(cls, config_path: str) -> "ModelConfig":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "model" not in config:
            raise ValueError("model section not found in config")
        return cls(**config["model"])
