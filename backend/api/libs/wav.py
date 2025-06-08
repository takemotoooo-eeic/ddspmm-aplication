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

def extract_loudness(
    signal: np.ndarray, block_size: int, n_fft: int = 2048
) -> np.ndarray:
    s: np.ndarray = librosa.stft(
        y=signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
    )
    s = np.abs(s)
    s = np.log10(np.maximum(1e-20, s)) * 20
    s = np.mean(s, axis=0)[..., :-1]
    return s


def extract_pitch(
    signal: np.ndarray, sampling_rate: int, block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    result = crepe.predict(
        audio=signal,
        sr=sampling_rate,
        step_size=int(1000 * block_size / sampling_rate),
        verbose=1,
        center=True,
        viterbi=True,
    )
    f0: np.ndarray = result[1].reshape(-1)[:-1]
    confidence: np.ndarray = result[2].reshape(-1)[:-1]

    length: int = signal.shape[-1] // block_size
    if f0.shape[-1] != length:
        f0 = np.interp(
            np.linspace(0, 1, length, endpoint=False),
            np.linspace(0, 1, f0.shape[-1], endpoint=False),
            f0,
        )
        confidence = np.interp(
            np.linspace(0, 1, length, endpoint=False),
            np.linspace(0, 1, confidence.shape[-1], endpoint=False),
            confidence,
        )

    return f0, confidence

def preprocess_wav_file(
    wav_file: bytes, preprocess_config: PreprocessConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    signal, _ = librosa.load(BytesIO(wav_file), sr=preprocess_config.sampling_rate)
    N: int = (
        preprocess_config.signal_length - len(signal) % preprocess_config.signal_length
    ) % preprocess_config.signal_length
    signal: np.ndarray = np.pad(signal, (0, N))

    pitch, _ = extract_pitch(
        signal, preprocess_config.sampling_rate, preprocess_config.block_size
    )
    loudness: np.ndarray = extract_loudness(signal, preprocess_config.block_size)
    signal = torch.from_numpy(np.array(signal)).to(device).float()
    pitch = torch.from_numpy(np.array(pitch)).to(device).float()
    loudness = torch.from_numpy(np.array(loudness)).to(device).float()
    return signal, pitch, loudness


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
