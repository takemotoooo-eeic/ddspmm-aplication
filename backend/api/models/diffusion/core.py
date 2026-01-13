import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import librosa as li
import crepe
import math
import torchaudio
from os import path, makedirs
import logging
import json
from typing import Any
import matplotlib

matplotlib.use("Agg")  # バックエンドを設定（GUI不要）
import matplotlib.pyplot as plt


def safe_log(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return torch.log(x + eps)


def multiscale_fft(
    signal: torch.Tensor, scales: list[int], overlap: float
) -> list[torch.Tensor]:
    stfts: list[torch.Tensor] = []
    for s in scales:
        S = torch.stft(
            input=signal,
            n_fft=s,
            hop_length=int(s * (1 - overlap)),
            win_length=s,
            window=torch.hann_window(s).to(signal),
            center=True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


def multiscale_fft_v2(
    signal: torch.Tensor, scales: list[int], overlap: float
) -> list[torch.Tensor]:
    stfts: list[torch.Tensor] = []
    for s in scales:
        # Flat top window using scipy
        from scipy.signal import windows

        flat_top_window = torch.from_numpy(windows.flattop(s)).to(
            signal.device, dtype=signal.dtype
        )

        S = torch.stft(
            input=signal,
            n_fft=s,
            hop_length=int(s * (1 - overlap)),
            win_length=s,
            window=flat_top_window,
            center=True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, scale_factor=factor, mode="linear")
    return signal.permute(0, 2, 1)


def remove_above_nyquist(amplitudes, pitch, sampling_rate):
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
    aa = (pitches < sampling_rate / 2).float() + 1e-4
    return amplitudes * aa


def scale_function(x):
    return 2 * torch.sigmoid(x) ** (math.log(10)) + 1e-7


def extract_loudness(
    signal: np.ndarray, block_size: int, n_fft: int = 2048
) -> np.ndarray:
    s: np.ndarray = li.stft(
        y=signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
    )
    s = np.abs(s)
    s = np.log10(np.maximum(1e-5, s)) * 20
    s = np.mean(s, axis=0)[..., :-1]
    return s


def extract_pitch(
    signal: np.ndarray, sampling_rate: int, block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    result = crepe.predict(
        audio=signal,
        sr=sampling_rate,
        step_size=int(1000 * block_size / sampling_rate),
        verbose=0,
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


def calculate_mfcc(audio: np.ndarray, sampling_rate: int) -> np.ndarray:
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sampling_rate,
        n_mfcc=30,
        log_mels=True,
        melkwargs=dict(
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=20.0,
            f_max=8000.0,
        ),
    )
    audio = torch.from_numpy(audio).unsqueeze(0)
    mfcc: torch.Tensor = mfcc_transform(audio)
    mfcc_np = mfcc.numpy()
    return mfcc_np


def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)


def gru(n_input, hidden_size):
    return nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)


def harmonic_synth(pitch, amplitudes, sampling_rate):
    n_harmonic = amplitudes.shape[-1]
    omega = torch.cumsum(2 * math.pi * pitch / sampling_rate, 1)
    omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal


def amp_to_impulse_response(amp, target_size):
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal, kernel):
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2 :]

    return output


# ============================================================================
# Diffusion用ヘルパー関数
# ============================================================================

def normalize(value: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """valueを正規化"""
    return (value - mean) / std


def denormalize(value: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """valueの正規化を解除"""
    return value * std + mean


def aggregate_params(
    f0: torch.Tensor,
    loudness: torch.Tensor,
    z_feature: torch.Tensor,
    loudness_mean: float,
    loudness_std: float,
    f0_mean: float,
    f0_std: float,
    z_feature_mean: torch.Tensor,
    z_feature_std: torch.Tensor,
) -> torch.Tensor:
    """
    合成パラメータを集約

    Args:
        f0: (B, L) - 基本周波数（またはf0_diff）
        loudness: (B, L) - ラウドネス（またはloudness_diff）
        z_feature: (B, L, 16) - z特徴量
        loudness_mean: loudness（またはloudness_diff）の平均
        loudness_std: loudness（またはloudness_diff）の標準偏差
        f0_mean: f0（またはf0_diff）の平均
        f0_std: f0（またはf0_diff）の標準偏差
        z_feature_mean: (1, 1, 16) - z_featureの平均
        z_feature_std: (1, 1, 16) - z_featureの標準偏差
    Returns:
        aggregated: (B, L, 18) - 集約された合成パラメータ
    """
    # loudnessを正規化
    loudness_normalized = normalize(loudness, loudness_mean, loudness_std)

    # f0を正規化（平均・分散で標準化）
    f0_normalized = normalize(f0, f0_mean, f0_std)

    # z_featureを正規化（データセット全体の統計情報を使用）
    z_feature_normalized = normalize(z_feature, z_feature_mean, z_feature_std)

    # 結合: (B, L, 18) = (B, L, 1) + (B, L, 1) + (B, L, 16)
    f0_expanded = f0_normalized.unsqueeze(-1)  # (B, L, 1)
    loudness_expanded = loudness_normalized.unsqueeze(-1)  # (B, L, 1)

    aggregated = torch.cat(
        [f0_expanded, loudness_expanded, z_feature_normalized], dim=-1
    )

    return aggregated


def split_params(
    aggregated: torch.Tensor,
    f0_mean: float,
    f0_std: float,
    z_feature_mean: torch.Tensor,
    z_feature_std: torch.Tensor,
    loudness_diff_mean: float,
    loudness_diff_std: float,
    loudness_mean: float,
    loudness_std: float,
    loudness_score: torch.Tensor | None = None,
    use_loudness_diff: bool = False,
    f0_diff_mean: float | None = None,
    f0_diff_std: float | None = None,
    f0_score: torch.Tensor | None = None,
    use_f0_diff: bool = False,
    allowed_f0_range: float | None = None,
    allowed_loudness_range: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    集約された合成パラメータを分割

    Args:
        aggregated: (B, L, 18) - 集約された合成パラメータ
        f0_mean: f0（またはf0_diff）の平均
        f0_std: f0（またはf0_diff）の標準偏差
        z_feature_mean: (1, 1, 16) - z_featureの平均（denormalize用）
        z_feature_std: (1, 1, 16) - z_featureの標準偏差（denormalize用）
        loudness_diff_mean: loudness_diffの平均（denormalize用）
        loudness_diff_std: loudness_diffの標準偏差（denormalize用）
        loudness_mean: loudnessの平均（normalize用）
        loudness_std: loudnessの標準偏差（normalize用）
        loudness_score: (B, L) - loudness_score（loudness_diff使用時に元に戻すため）
        use_loudness_diff: loudness_diffを使用しているかどうか
        f0_diff_mean: f0_diffの平均（denormalize用、use_f0_diffがTrueの場合）
        f0_diff_std: f0_diffの標準偏差（denormalize用、use_f0_diffがTrueの場合）
        f0_score: (B, L) - f0_score（f0_diff使用時に元に戻すため）
        use_f0_diff: f0_diffを使用しているかどうか

    Returns:
        f0: (B, L) - 基本周波数（denormalize済み）
        loudness_normalized: (B, L) - ラウドネス（正規化済み、DDSPに入力可能な形式）
        z_feature: (B, L, 16) - z特徴量（denormalize済み）
    """
    f0_normalized = aggregated[:, :, 0]  # (B, L)
    loudness_normalized = aggregated[:, :, 1]  # (B, L)
    z_feature_normalized = aggregated[:, :, 2:]  # (B, L, 16)

    # f0の処理
    if use_f0_diff:
        f0_diff = denormalize(f0_normalized, f0_diff_mean, f0_diff_std)
        if allowed_f0_range is not None:
            f0_diff = torch.clamp(f0_diff, -allowed_f0_range, allowed_f0_range)
        f0 = f0_diff + f0_score
    else:
        f0 = denormalize(f0_normalized, f0_mean, f0_std)
        
    f0 = torch.clamp(f0, -10000.0, 10000.0)

    # z_featureをdenormalize
    z_feature = denormalize(z_feature_normalized, z_feature_mean, z_feature_std)

    # loudnessの処理
    if use_loudness_diff:
        loudness_diff = denormalize(loudness_normalized, loudness_diff_mean, loudness_diff_std)
        if allowed_loudness_range is not None:
            loudness_diff = torch.clamp(loudness_diff, -allowed_loudness_range, allowed_loudness_range)
        loudness = loudness_diff + loudness_score
    return f0, f0_diff, loudness, loudness_diff, z_feature
