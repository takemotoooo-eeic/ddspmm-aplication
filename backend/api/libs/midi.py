import os

import numpy as np
import torch
from fastapi import UploadFile

from api.libs.exceptions import BadRequest
from api.models.midi_aligner.midi_aligner import AlignedMidi


def verify_mid_file_format(file: UploadFile) -> None:
    if not file or not file.filename:
        raise BadRequest("ファイルがアップロードされていません。")
    if not file.filename.endswith(".mid"):
        raise BadRequest("ファイルはmid形式である必要があります。")
    try:
        _, ext = os.path.splitext(file.filename)
    except Exception:
        raise BadRequest("ファイルの拡張子が取得できません。")
    if ext != ".mid":
        raise BadRequest("ファイルはmid形式である必要があります。")


def convert_midi_to_features(
    midi: AlignedMidi,
    loudness: torch.Tensor,
    sampling_rate: int,
    signal_length: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pitch_array = np.zeros(signal_length)

    for note in midi.notes:
        start_time = float(note.start)
        duration = float(note.duration)
        pitch = float(note.frequency)

        start_sample = int(start_time * sampling_rate)
        end_sample = int((start_time + duration) * sampling_rate)

        if start_sample < signal_length:
            end_sample = min(end_sample, signal_length)
            pitch_array[start_sample:end_sample] = pitch

    pitch_array = pitch_array[::block_size]

    loudness[pitch_array == 0] = -60

    mask: torch.Tensor = torch.from_numpy(pitch_array).float().to(device)
    mask[mask != 0] = 1

    pitch_array[pitch_array == 0] = np.mean(pitch_array[pitch_array != 0])

    return (
        torch.from_numpy(pitch_array).float().to(device),
        loudness.float().to(device),
        mask.to(device),
    )
