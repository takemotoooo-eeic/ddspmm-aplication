import os

import numpy as np
import torch
from fastapi import UploadFile

from api.libs.exceptions import BadRequest
from api.libs.note import AlignedMidi
from api.libs.const import URMP_LOUDNESS_SCORE_SILENCE, URMP_LOUDNESS_SCORE_LOUD


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
    sampling_rate: int,
    signal_length: int,
    device: torch.device,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    sorted_notes = sorted(midi.notes, key=lambda n: n.start)
    mean_frequency = np.mean([note.frequency for note in midi.notes])
    pitch_array = np.full(signal_length, mean_frequency)
    loudness_array = np.full(signal_length, URMP_LOUDNESS_SCORE_SILENCE)


    for i, note in enumerate(sorted_notes):
        start_sample = int(note.start * sampling_rate)
        end_sample = int((note.start + note.duration) * sampling_rate)

        if start_sample < signal_length:
            end_sample = min(end_sample, signal_length)
            
            # 現在の音符の開始位置まで、直前の音符のf0で埋める
            if i == 0:
                # 最初の音符の場合、開始位置まで平均値で埋める
                pitch_array[:start_sample] = mean_frequency
            else:
                # 直前の音符のf0で埋める
                pitch_array[last_end_sample:start_sample] = last_frequency
            
            # 現在の音符の範囲を設定
            pitch_array[start_sample:end_sample] = note.frequency
            loudness_array[start_sample:end_sample] = URMP_LOUDNESS_SCORE_LOUD
            
            last_frequency = note.frequency
            last_end_sample = end_sample
    
    # 最後の音符以降も直前の音符のf0で埋める
    if len(sorted_notes) > 0 and last_end_sample < signal_length:
        pitch_array[last_end_sample:] = last_frequency

    return (
        torch.from_numpy(pitch_array).float().to(device)[::block_size],
        torch.from_numpy(loudness_array).float().to(device)[::block_size],
    )
