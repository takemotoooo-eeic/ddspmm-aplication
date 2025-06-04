import torch

from api.config import ModelConfig
from api.models.ddsp.model import DDSP


def convert_weights(old_state_dict):
    """古い重みファイルを新しいモデル構造に変換する"""
    new_state_dict = {}

    # 共通のパラメータ
    common_params = ["sampling_rate", "block_size", "cache_gru", "phase"]
    for param in common_params:
        if param in old_state_dict:
            new_state_dict[param] = old_state_dict[param]

    # MLPパラメータの変換
    for i in range(3):  # 3つのMLP
        for j in range(8):  # 各MLPの層
            old_key = f"in_mlps.{i}.{j}.weight"
            if old_key in old_state_dict:
                new_state_dict[f"decoder.in_mlps.{i}.{j}.weight"] = old_state_dict[
                    old_key
                ]
            old_key = f"in_mlps.{i}.{j}.bias"
            if old_key in old_state_dict:
                new_state_dict[f"decoder.in_mlps.{i}.{j}.bias"] = old_state_dict[
                    old_key
                ]

    # GRUパラメータの変換
    gru_params = ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"]
    for param in gru_params:
        old_key = f"gru.{param}"
        if old_key in old_state_dict:
            new_state_dict[f"decoder.gru.{param}"] = old_state_dict[old_key]

    # 出力MLPパラメータの変換
    for i in range(8):  # 出力MLPの層
        old_key = f"out_mlp.{i}.weight"
        if old_key in old_state_dict:
            new_state_dict[f"decoder.out_mlp.{i}.weight"] = old_state_dict[old_key]
        old_key = f"out_mlp.{i}.bias"
        if old_key in old_state_dict:
            new_state_dict[f"decoder.out_mlp.{i}.bias"] = old_state_dict[old_key]

    # 投影行列の変換
    for i in range(2):
        old_key = f"proj_matrices.{i}.weight"
        if old_key in old_state_dict:
            new_state_dict[f"decoder.proj_matrices.{i}.weight"] = old_state_dict[
                old_key
            ]
        old_key = f"proj_matrices.{i}.bias"
        if old_key in old_state_dict:
            new_state_dict[f"decoder.proj_matrices.{i}.bias"] = old_state_dict[old_key]

    # Reverbパラメータの変換
    reverb_params = ["noise", "decay", "wet", "t"]
    for param in reverb_params:
        old_key = f"reverb.{param}"
        if old_key in old_state_dict:
            new_state_dict[f"decoder.reverb.{param}"] = old_state_dict[old_key]

    return new_state_dict


def load_model(
    model_path: str, device: torch.device, model_config: ModelConfig
) -> DDSP:
    print(f"device: {device}")

    model = DDSP(
        hidden_size=model_config.hidden_size,
        n_harmonic=model_config.n_harmonic,
        n_bands=model_config.n_bands,
        sampling_rate=model_config.sampling_rate,
        block_size=model_config.block_size,
    ).to(device)

    # 重みファイルの読み込みと変換
    old_state_dict = torch.load(model_path, map_location=device)
    new_state_dict = convert_weights(old_state_dict)

    # 変換した重みをロード
    model.load_state_dict(new_state_dict, strict=False)

    print(f"model loaded: {model_path}")
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")
    return model
