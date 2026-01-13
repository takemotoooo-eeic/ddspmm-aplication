from pydantic import BaseModel, StrictInt, StrictFloat, StrictBool, StrictStr
from enum import Enum
import yaml


class ModelType(str, Enum):
    UNET = "unet"
    DIT = "dit"



class DiffusionModelConfig(BaseModel):
    model_type: StrictStr = "unet"  # "unet" or "dit"
    
    in_dim: StrictInt = 18  # f0(1) + loudness(1) + z_feature(16)
    time_embed_dim: StrictInt = 128
    film_cond_dim: StrictInt = 64

    # Conditioning設定
    use_film: StrictBool = True  # FiLMによる条件付けを使用
    use_cross_attention: StrictBool = False  # Cross-attentionによる条件付けを使用
    cfg_dropout_prob: StrictFloat = 0.1  # CFG用のdropout確率（学習時に楽器ラベルを無効化する確率、0.0-1.0）
    f0_score_scale: StrictFloat = 1.0  # f0_scoreのadditive conditioningのスケーリング係数（0.0-1.0、小さいほどf0_scoreの影響が弱い）
    use_note_mask: StrictBool = False  # note_maskによる条件付けを使用
    note_mask_scale: StrictFloat = 1.0  # note_maskのadditive conditioningのスケーリング係数（0.0-1.0、小さいほどnote_maskの影響が弱い）

    # UNet構造パラメータ
    down1_out_ch: StrictInt = 64
    down2_out_ch: StrictInt = 128
    down3_out_ch: StrictInt = 256
    bot1_out_ch: StrictInt = 512

    # DiT構造パラメータ
    dit_hidden_dim: StrictInt = 768  # Transformer hidden dimension
    dit_num_layers: StrictInt = 12  # Number of Transformer layers
    dit_num_heads: StrictInt = 12  # Number of attention heads
    dit_mlp_ratio: StrictFloat = 4.0  # MLP expansion ratio

    # Diffusionパラメータ
    num_timesteps: StrictInt = 1000
    beta_start: StrictFloat = 0.0001
    beta_end: StrictFloat = 0.02
    use_diffusers: StrictBool = False  # diffusersライブラリを使用するか

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
