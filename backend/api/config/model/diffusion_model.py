from pydantic import BaseModel, StrictInt, StrictFloat, StrictBool, StrictStr
import yaml

class DiffusionModelConfig(BaseModel):
    time_embed_dim: StrictInt = 128
    film_cond_dim: StrictInt = 64

    # Conditioning設定
    cfg_dropout_prob: StrictFloat = 0.1
    f0_score_scale: StrictFloat = 1.0
    cross_attention_dim: StrictInt | None = 128
    conditioning_type: StrictStr = "cross_attention"
    add_input_mode: StrictStr = "add"

    # UNet構造パラメータ
    down1_out_ch: StrictInt = 64
    down2_out_ch: StrictInt = 128
    down3_out_ch: StrictInt = 256
    bot1_out_ch: StrictInt = 512
    
    # Diffusionパラメータ
    num_timesteps: StrictInt = 1000
    beta_start: StrictFloat = 0.0001
    beta_end: StrictFloat = 0.02


    @classmethod
    def from_config_path(cls, config_path: str) -> "DiffusionModelConfig":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "diffusion_model" not in config:
            raise ValueError("diffusion_model section not found in config")
        # _target_フィールドを削除（Hydraの設定ファイルから読み込む場合に含まれる）
        diffusion_model_dict = config["diffusion_model"].copy()
        if "_target_" in diffusion_model_dict:
            del diffusion_model_dict["_target_"]
        return cls(**diffusion_model_dict)
