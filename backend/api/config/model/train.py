from pydantic import BaseModel, StrictInt, StrictFloat, StrictBool, StrictStr
import yaml


class TrainConfig(BaseModel):
    model_dir: StrictStr
    statistics_dir: StrictStr
    batch_size: StrictInt
    epochs: StrictInt
    lr: StrictFloat
    debug: StrictBool
    output_dir: StrictStr

    diffusion_model_dir: StrictStr | None = None
    guidance_scale_start: StrictFloat | None = None
    guidance_scale_end: StrictFloat | None = None
    enable_guidance: StrictBool = False

    direct_optim: list[StrictStr] | None = None # ["pitch", "loudness", "z_feature"]
    guiding_params: list[StrictStr] = ["pitch", "loudness", "z_feature"] # ["pitch", "loudness", "z_feature"]

    @classmethod
    def from_config_path(cls, config_path: str) -> "TrainConfig":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "train" not in config:
            raise ValueError("train section not found in config")
        train_dict = dict(config["train"])
        train_dict.pop("_target_", None)
        return cls(**train_dict)
