import json
import os
import subprocess
import tempfile
from io import BytesIO
import librosa
import numpy as np
import pretty_midi
import soundfile as sf

from api.libs.note import Note


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


class FluidSynthRenderer:
    """FluidSynth CLI を用いて MIDI を WAV にレンダリングする。"""

    def __init__(
        self,
        soundfont_path: str,
        gm_program_mapping_path: str,
        sample_rate: int | None = None,
        gain: float | None = None,
        polyphony: int | None = None,
        sample_format: str | None = None,
        output_format: str | None = None,
    ):
        self.soundfont_path = soundfont_path
        self.sample_rate = 16000
        self.gain = gain if gain is not None else _env_float("GAIN", 0.21)
        self.polyphony = polyphony or _env_int("SCO_POLYPHONY", 512)
        self.sample_format = sample_format or os.environ.get("SCO_SAMPLE_FORMAT", "16bits")
        self.output_format = output_format or os.environ.get("SCO_OUTPUT_FORMAT", "s16")
        self.ncpu = _env_int("FLUIDSYNTH_CPU_CORES", os.cpu_count() or 4)

        if not os.path.isfile(self.soundfont_path):
            raise FileNotFoundError(
                f"SoundFont not found: {self.soundfont_path}. "
                "Place FluidR3_GM.sf2 under api/models/fluidsynth_assets/soundfonts/ "
                "(run: ./scripts/download_soundfont.sh)."
            )

        with open(gm_program_mapping_path, "r") as f:
            self.gm_program_mapping: dict[str, int] = json.load(f)

    def get_gm_program(self, instrument_name: str) -> int:
        code = instrument_name.lower()
        if code not in self.gm_program_mapping:
            raise ValueError(
                f"Instrument '{instrument_name}' not found in GM mapping. "
                f"Available: {sorted(self.gm_program_mapping.keys())}"
            )
        return self.gm_program_mapping[code]

    def render_midi_bytes(self, midi_bytes: bytes) -> bytes:
        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = os.path.join(tmpdir, "input.mid")
            wav_path = os.path.join(tmpdir, "output.wav")
            with open(midi_path, "wb") as f:
                f.write(midi_bytes)
            self._run_fluidsynth(midi_path, wav_path)
            with open(wav_path, "rb") as f:
                return f.read()

    def render_midi_to_samples(
        self,
        midi_bytes: bytes,
        target_length: int | None = None,
        target_sample_rate: int | None = None,
    ) -> np.ndarray:
        wav_bytes = self.render_midi_bytes(midi_bytes)
        audio, sr = sf.read(BytesIO(wav_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        out_sr = target_sample_rate or self.sample_rate
        if sr != out_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=out_sr)

        if target_length is not None:
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            elif len(audio) > target_length:
                audio = audio[:target_length]
        return audio

    def notes_to_midi_bytes(
        self,
        notes: list[Note],
        program: int,
        total_duration_sec: float,
    ) -> bytes:
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=program)
        for note in notes:
            midi_pitch = int(round(12 * np.log2(max(note.frequency, 1e-7) / 440.0) + 69))
            midi_pitch = max(0, min(127, midi_pitch))
            end = min(note.start + note.duration, total_duration_sec)
            if end <= note.start:
                continue
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=midi_pitch,
                    start=max(0.0, note.start),
                    end=end,
                )
            )
        pm.instruments.append(instrument)
        buffer = BytesIO()
        pm.write(buffer)
        return buffer.getvalue()

    def split_midi_by_instrument(self, midi_bytes: bytes) -> list[tuple[str, bytes]]:
        midi_buffer = BytesIO(midi_bytes)
        midi_buffer.seek(0)
        midi_data = pretty_midi.PrettyMIDI(midi_buffer)

        results: list[tuple[str, bytes]] = []
        instrument_name_count: dict[str, int] = {}

        for track in midi_data.instruments:
            if not track.notes:
                continue

            base_name = pretty_midi.program_to_instrument_name(track.program)
            if base_name in instrument_name_count:
                instrument_name_count[base_name] += 1
                instrument_name = f"{base_name}_{instrument_name_count[base_name]}"
            else:
                instrument_name_count[base_name] = 0
                instrument_name = base_name

            single_pm = pretty_midi.PrettyMIDI()
            single_track = pretty_midi.Instrument(
                program=track.program,
                is_drum=track.is_drum,
                name=track.name,
            )
            single_track.notes = list(track.notes)
            single_pm.instruments.append(single_track)

            out_buffer = BytesIO()
            single_pm.write(out_buffer)
            results.append((instrument_name, out_buffer.getvalue()))

        return results

    def _run_fluidsynth(self, midi_path: str, wav_path: str) -> None:
        cmd = [
            "fluidsynth",
            "-ni",
            "-q",
            "-g",
            str(self.gain),
            "-R",
            "0",
            "-C",
            "0",
            "-r",
            str(self.sample_rate),
            "-o",
            f"synth.cpu-cores={self.ncpu}",
            "-o",
            f"synth.polyphony={self.polyphony}",
            "-o",
            f"synth.sample-rate={self.sample_rate}",
            "-o",
            f"audio.sample-format={self.sample_format}",
            "-O",
            self.output_format,
            "-F",
            wav_path,
            self.soundfont_path,
            midi_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise RuntimeError(
                "fluidsynth command not found. Install fluidsynth in the environment."
            ) from e
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.strip() if e.stderr else ""
            raise RuntimeError(f"FluidSynth rendering failed: {stderr}") from e
