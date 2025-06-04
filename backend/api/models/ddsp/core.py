import math

import crepe
import librosa as li
import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn


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


def resample(x, factor: int):
    batch, frame, channel = x.shape
    x = x.permute(0, 2, 1).reshape(batch * channel, 1, frame)

    window = torch.hann_window(
        factor * 2,
        dtype=x.dtype,
        device=x.device,
    ).reshape(1, 1, -1)
    y = torch.zeros(x.shape[0], x.shape[1], factor * x.shape[2]).to(x)
    y[..., ::factor] = x
    y[..., -1:] = x[..., -1:]
    y = torch.nn.functional.pad(y, [factor, factor])
    y = torch.nn.functional.conv1d(y, window)[..., :-1]

    y = y.reshape(batch, channel, factor * frame).permute(0, 2, 1)

    return y


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
