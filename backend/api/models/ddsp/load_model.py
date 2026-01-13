import torch

from api.config import ModelConfig
from api.models.ddsp.model import DDSP


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

    # 変換した重みをロード
    model.load_state_dict(old_state_dict, strict=False)

    print(f"model loaded: {model_path}")
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")
    return model
