from pydantic import BaseModel, StrictInt, StrictBool, StrictStr
import yaml
from enum import Enum


class DataDomain(str, Enum):
    ENSEMBLE_SET = "ensemble_set"
    URMP_SET = "urmp_dataset"


class PreprocessConfig(BaseModel):
    domain: DataDomain
    data_location: StrictStr
    sampling_rate: StrictInt
    signal_length: StrictInt
    block_size: StrictInt
    oneshot: StrictBool
    out_dir: StrictStr

    @classmethod
    def from_config_path(cls, config_path: str) -> "PreprocessConfig":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "preprocess" in config:
            data = dict(config["preprocess"])
            if "domain" not in data:
                data["domain"] = DataDomain.URMP_SET
            return cls(**data)
        return cls(**config)
