import io
from typing import Tuple

import numpy as np
import soundfile as sf

MELODYFLOW_MAX_SEGMENT_SEC = 30.0

# MelodyFlow 出力のゲイン調整（元区間との RMS 比）
DEFAULT_MIN_LOUDNESS_GAIN = 0.25
DEFAULT_MAX_LOUDNESS_GAIN = 4.0
SILENCE_RMS_THRESHOLD = 1e-6


def read_wav_bytes(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """WAV bytes -> (samples, channels) float array and sample rate."""
    data, sr = sf.read(io.BytesIO(wav_bytes), always_2d=True)
    return data.astype(np.float32), int(sr)


def write_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _rms(audio: np.ndarray) -> float:
    flat = np.asarray(audio, dtype=np.float64).ravel()
    if flat.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(flat * flat)))


def match_segment_loudness(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    min_gain: float = DEFAULT_MIN_LOUDNESS_GAIN,
    max_gain: float = DEFAULT_MAX_LOUDNESS_GAIN,
    silence_rms: float = SILENCE_RMS_THRESHOLD,
) -> np.ndarray:
    """
    編集後セグメントの RMS を参照セグメントに合わせる。
    極端なゲイン変化を避けるため min_gain〜max_gain でクランプする。
    """
    ref_rms = _rms(reference)
    tgt_rms = _rms(target)
    if ref_rms < silence_rms or tgt_rms < silence_rms:
        return target.astype(np.float32, copy=False)

    gain = float(np.clip(ref_rms / tgt_rms, min_gain, max_gain))
    scaled = target.astype(np.float32) * gain

    peak = float(np.max(np.abs(scaled)))
    if peak > 1.0:
        scaled = scaled * (0.99 / peak)

    return scaled


def resample_to_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    if audio.shape[0] == target_len:
        return audio
    x_old = np.linspace(0.0, 1.0, audio.shape[0])
    x_new = np.linspace(0.0, 1.0, target_len)
    if audio.ndim == 1:
        return np.interp(x_new, x_old, audio).astype(np.float32)
    channels = audio.shape[1]
    out = np.zeros((target_len, channels), dtype=np.float32)
    for c in range(channels):
        out[:, c] = np.interp(x_new, x_old, audio[:, c])
    return out


def validate_segment(
    duration_sec: float,
    start_sec: float,
    end_sec: float,
    max_segment_sec: float = MELODYFLOW_MAX_SEGMENT_SEC,
) -> None:
    if start_sec < 0:
        raise ValueError("start_sec は 0 以上である必要があります。")
    if end_sec <= start_sec:
        raise ValueError("end_sec は start_sec より大きい必要があります。")
    if end_sec > duration_sec:
        raise ValueError("end_sec が音声の長さを超えています。")
    segment_len = end_sec - start_sec
    if segment_len > max_segment_sec:
        raise ValueError(
            f"編集区間は最大 {max_segment_sec} 秒です（指定: {segment_len:.2f} 秒）。"
        )


def splice_region(
    full: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    replacement: np.ndarray,
) -> np.ndarray:
    """Replace [start_sec, end_sec) with replacement (resampled to segment length)."""
    start = int(round(start_sec * sr))
    end = int(round(end_sec * sr))
    seg_len = end - start
    if seg_len <= 0:
        raise ValueError("編集区間が空です。")

    if replacement.ndim == 1:
        replacement = replacement[:, np.newaxis]
    if full.ndim == 1:
        full = full[:, np.newaxis]

    if replacement.shape[1] != full.shape[1]:
        if replacement.shape[1] == 1 and full.shape[1] > 1:
            replacement = np.tile(replacement, (1, full.shape[1]))
        elif full.shape[1] == 1 and replacement.shape[1] > 1:
            replacement = replacement[:, :1]

    replacement = resample_to_length(replacement, seg_len)
    result = full.copy()
    result[start:end, :] = replacement
    if result.shape[1] == 1:
        return result[:, 0]
    return result
