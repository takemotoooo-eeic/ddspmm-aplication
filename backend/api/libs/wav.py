import os

import crepe
import librosa
import numpy as np
import torch
from fastapi import UploadFile

from api.config import PreprocessConfig
from api.libs.exceptions import BadRequest
from io import BytesIO


def verify_wav_file_format(file: UploadFile) -> None:
    if not file or not file.filename:
        raise BadRequest("ファイルがアップロードされていません。")
    if not file.filename.endswith(".wav"):
        raise BadRequest("ファイルはwav形式である必要があります。")
    try:
        _, ext = os.path.splitext(file.filename)
    except Exception:
        raise BadRequest("ファイルの拡張子が取得できません。")
    if ext != ".wav":
        raise BadRequest("ファイルはwav形式である必要があります。")

def preprocess_wav_file(
    wav_file: bytes, preprocess_config: PreprocessConfig, device: torch.device
) -> torch.Tensor:
    signal, _ = librosa.load(BytesIO(wav_file), sr=preprocess_config.sampling_rate)
    N: int = (
        preprocess_config.signal_length - len(signal) % preprocess_config.signal_length
    ) % preprocess_config.signal_length
    signal: np.ndarray = np.pad(signal, (0, N))

    signal = torch.from_numpy(np.array(signal)).to(device).float()
    return signal


def reshape_to_segments(
    input: dict[str, torch.Tensor], signal_length: int
) -> dict[str, torch.Tensor]:
    if input.get("signal") is None:
        raise ValueError("signal must be in input")
    signal = input["signal"]
    signal = signal.reshape(-1, signal_length)
    segment_num = signal.shape[0]
    result: dict[str, torch.Tensor] = {}
    for key, value in input.items():
        if key == "signal":
            continue
        if key == "z_feature":
            value = value.reshape(segment_num, -1, 16)
        else:
            value = value.reshape(segment_num, -1)
        result[key] = value
    result["signal"] = signal
    return result
