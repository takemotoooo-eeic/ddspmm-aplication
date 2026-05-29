import io
import json
import os
import zipfile
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from pydantic import BaseModel

from api.libs.const import (
    DEFAULT_SAMPLING_RATE,
    FLUIDSYNTH_GM_PROGRAM_MAPPING_PATH,
    FLUIDSYNTH_SOUNDFONT_PATH,
)
from api.libs.fluidsynth_render import FluidSynthRenderer
from api.libs.logging import get_logger
from api.libs.note import AlignedMidi, Note


class FluidSynthGenerateParams(BaseModel):
    notes: list[Note]
    instrument_name: str
    signal_length: int


@dataclass
class FluidSynthTrainInput:
    aligned_midi_list: list[AlignedMidi]
    instrument_names: list[str]


@dataclass
class FluidSynthTrainResult:
    instrument_name: str
    wav_bytes: bytes


class FluidSynthModel:
    def __init__(self, output_sample_rate: int = DEFAULT_SAMPLING_RATE):
        self.logger = get_logger()
        self.output_sample_rate = output_sample_rate
        soundfont_path = os.environ.get("SOUNDFONT", FLUIDSYNTH_SOUNDFONT_PATH)
        self.renderer = FluidSynthRenderer(
            soundfont_path=soundfont_path,
            gm_program_mapping_path=FLUIDSYNTH_GM_PROGRAM_MAPPING_PATH,
        )

    def train(self, train_input: FluidSynthTrainInput) -> list[FluidSynthTrainResult]:
        """アライン済み MIDI の各楽器を FluidSynth でレンダリングする。"""
        if len(train_input.aligned_midi_list) != len(train_input.instrument_names):
            raise ValueError(
                "アライン済みMIDI数と楽器名数が一致しません。"
                f" (midi: {len(train_input.aligned_midi_list)}, names: {len(train_input.instrument_names)})"
            )
        if not train_input.aligned_midi_list:
            raise ValueError("アライン済みMIDIがありません。")

        results: list[FluidSynthTrainResult] = []
        for aligned_midi, instrument_name in zip(
            train_input.aligned_midi_list, train_input.instrument_names
        ):
            if not aligned_midi.notes:
                raise ValueError(f"楽器 '{instrument_name}' に音符がありません。")

            program = self.renderer.get_gm_program(instrument_name)
            total_duration_sec = max(
                note.start + note.duration for note in aligned_midi.notes
            )
            midi_bytes = self.renderer.notes_to_midi_bytes(
                notes=aligned_midi.notes,
                program=program,
                total_duration_sec=total_duration_sec,
            )
            self.logger.info(f"Rendering FluidSynth track: {instrument_name}")
            target_length = int(total_duration_sec * self.output_sample_rate)
            audio = self.renderer.render_midi_to_samples(
                midi_bytes,
                target_length=target_length,
                target_sample_rate=self.output_sample_rate,
            )
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio, self.output_sample_rate, format="WAV")
            wav_buffer.seek(0)
            wav_bytes = wav_buffer.read()
            results.append(
                FluidSynthTrainResult(
                    instrument_name=instrument_name,
                    wav_bytes=wav_bytes,
                )
            )
        return results

    @staticmethod
    def _safe_wav_stem(instrument_name: str) -> str:
        return "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in instrument_name
        )

    def train_to_zip(self, train_input: FluidSynthTrainInput) -> bytes:
        results = self.train(train_input)
        buffer = io.BytesIO()
        manifest_features = []
        used_stems: dict[str, int] = {}
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item, aligned_midi in zip(results, train_input.aligned_midi_list):
                base_stem = self._safe_wav_stem(item.instrument_name)
                if base_stem in used_stems:
                    used_stems[base_stem] += 1
                    wav_stem = f"{base_stem}_{used_stems[base_stem]}"
                else:
                    used_stems[base_stem] = 0
                    wav_stem = base_stem
                zf.writestr(f"{wav_stem}.wav", item.wav_bytes)
                manifest_features.append(
                    {
                        "instrument_name": item.instrument_name,
                        "wav_file": f"{wav_stem}.wav",
                        "notes": [
                            {
                                "start": note.start,
                                "frequency": note.frequency,
                                "duration": note.duration,
                            }
                            for note in aligned_midi.notes
                        ],
                    }
                )
            zf.writestr(
                "manifest.json",
                json.dumps({"features": manifest_features}, ensure_ascii=False),
            )
        buffer.seek(0)
        return buffer.getvalue()

    def generate(self, params: FluidSynthGenerateParams) -> bytes:
        """音符列から MIDI を作成し、単一楽器の WAV を生成する。"""
        program = self.renderer.get_gm_program(params.instrument_name)
        total_duration_sec = params.signal_length / self.output_sample_rate
        midi_bytes = self.renderer.notes_to_midi_bytes(
            notes=params.notes,
            program=program,
            total_duration_sec=total_duration_sec,
        )
        audio = self.renderer.render_midi_to_samples(
            midi_bytes,
            target_length=params.signal_length,
            target_sample_rate=self.output_sample_rate,
        )
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, self.output_sample_rate, format="WAV")
        wav_buffer.seek(0)
        return wav_buffer.read()
