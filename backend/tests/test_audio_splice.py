import numpy as np
import pytest

from api.libs.audio_splice import (
    MELODYFLOW_MAX_SEGMENT_SEC,
    match_segment_loudness,
    read_wav_bytes,
    splice_region,
    validate_segment,
    write_wav_bytes,
)


def test_validate_segment_ok():
    validate_segment(10.0, 1.0, 3.0)


def test_validate_segment_rejects_invalid_range():
    with pytest.raises(ValueError, match="start_sec"):
        validate_segment(10.0, -0.1, 3.0)
    with pytest.raises(ValueError, match="end_sec"):
        validate_segment(10.0, 3.0, 2.0)
    with pytest.raises(ValueError, match="長さ"):
        validate_segment(10.0, 1.0, 11.0)
    with pytest.raises(ValueError, match="最大"):
        validate_segment(
            60.0,
            0.0,
            MELODYFLOW_MAX_SEGMENT_SEC + 1,
            max_segment_sec=MELODYFLOW_MAX_SEGMENT_SEC,
        )


def test_splice_region_replaces_middle():
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    full = np.sin(2 * np.pi * 440 * t).astype(np.float32)[:, np.newaxis]
    replacement = np.zeros((int(0.5 * sr), 1), dtype=np.float32)
    result = splice_region(full, sr, 0.5, 1.0, replacement)
    start = int(0.5 * sr)
    end = int(1.0 * sr)
    region = result[start:end, 0] if result.ndim > 1 else result[start:end]
    assert np.allclose(region, 0.0)
    before = result[:start, 0] if result.ndim > 1 else result[:start]
    assert not np.allclose(before, 0.0)


def test_match_segment_loudness_scales_to_reference():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    reference = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    target = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    matched = match_segment_loudness(reference, target)
    ref_rms = np.sqrt(np.mean(reference**2))
    matched_rms = np.sqrt(np.mean(matched**2))
    assert matched_rms == pytest.approx(ref_rms, rel=0.02)


def test_match_segment_loudness_clamps_extreme_gain():
    reference = np.ones(1000, dtype=np.float32) * 0.5
    target = np.ones(1000, dtype=np.float32) * 0.001
    matched = match_segment_loudness(reference, target, min_gain=0.25, max_gain=4.0)
    assert np.max(np.abs(matched)) <= 1.0


def test_wav_roundtrip():
    sr = 22050
    audio = (np.random.randn(sr * 2) * 0.1).astype(np.float32)
    wav_bytes = write_wav_bytes(audio, sr)
    loaded, loaded_sr = read_wav_bytes(wav_bytes)
    assert loaded_sr == sr
    assert loaded.shape[0] == audio.shape[0]
