import yaml
from pydantic import BaseModel


class TrainConfig(BaseModel):
    data_dir: str
    statistics_dir: str
    batch_size: int
    epochs: int
    lr: float
    debug: bool

    @classmethod
    def from_config_path(cls, config_path: str) -> "TrainConfig":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "train" not in config:
            raise ValueError("train section not found in config")
        return cls(**config["train"])
