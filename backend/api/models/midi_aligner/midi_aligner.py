from io import BytesIO

import librosa
import numpy as np
import pandas as pd
import pretty_midi
import scipy
from pydantic import BaseModel
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.dtw.utils import (
    compute_optimal_chroma_shift,
    make_path_strictly_monotonic,
    shift_chroma_vectors,
)
from synctoolbox.feature.chroma import (
    pitch_to_chroma,
    quantize_chroma,
    quantized_chroma_to_CENS,
)
from synctoolbox.feature.csv_tools import (
    df_to_pitch_features,
    df_to_pitch_onset_features,
)
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.utils import estimate_tuning

from api.libs.logging import get_logger


class Note(BaseModel):
    start: float
    frequency: float
    duration: float


class AlignedMidi(BaseModel):
    notes: list[Note]


class MidiAligner:
    def __init__(
        self,
        sr: float = 22050,
        hop: int = 512,
        feature_rate: int = 100,
        step_weight: np.ndarray = np.array([1.5, 1.5, 2.0]),
        threshold_rec: int = 10**7,
    ):
        self.sr: int = sr
        self.hop: int = hop
        self.feature_rate: float = feature_rate
        self.step_weight: np.ndarray = step_weight
        self.threshold_rec: int = threshold_rec
        self.trim_top_db: int = 20
        self.logger = get_logger()

    def _convert_midi_to_dataframe(self, midi_file: bytes) -> tuple[pd.DataFrame, list[str]]:
        midi_buffer = BytesIO(midi_file)
        midi_buffer.seek(0)
        midi_data = pretty_midi.PrettyMIDI(midi_buffer)
        rows = []
        instrument_name_count = {}
        instrument_names = []
        for instrument in midi_data.instruments:
            base_name = pretty_midi.program_to_instrument_name(instrument.program)
            if base_name in instrument_name_count:
                instrument_name_count[base_name] += 1
                instrument_name = f"{base_name}_{instrument_name_count[base_name]}"
            else:
                instrument_name_count[base_name] = 0
                instrument_name = base_name
            instrument_names.append(instrument_name)
            for note in instrument.notes:
                start = round(note.start, 6)
                duration = round(note.end - note.start, 6)
                pitch = note.pitch
                velocity = note.velocity
                rows.append([start, duration, pitch, velocity, instrument_name])

        df = pd.DataFrame(
            rows, columns=["start", "duration", "pitch", "velocity", "instrument"]
        )
        return df, instrument_names

    def _get_features_from_audio(
        self,
        audio: np.ndarray,
        tuning_offset: int,
        Fs: int,
        feature_rate: int,
        visualize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        f_pitch = audio_to_pitch_features(
            f_audio=audio,
            Fs=Fs,
            tuning_offset=tuning_offset,
            feature_rate=feature_rate,
            verbose=visualize,
        )
        f_chroma = pitch_to_chroma(f_pitch=f_pitch)
        f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)

        f_pitch_onset = audio_to_pitch_onset_features(
            f_audio=audio, Fs=Fs, tuning_offset=tuning_offset, verbose=visualize
        )
        f_DLNCO = pitch_onset_features_to_DLNCO(
            f_peaks=f_pitch_onset,
            feature_rate=feature_rate,
            feature_sequence_length=f_chroma_quantized.shape[1],
            visualize=visualize,
        )
        return f_chroma_quantized, f_DLNCO

    def _get_features_from_annotation(
        self, df_annotation: pd.DataFrame, feature_rate: int, visualize: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        f_pitch = df_to_pitch_features(df_annotation, feature_rate=feature_rate)
        f_chroma = pitch_to_chroma(f_pitch=f_pitch)
        f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
        f_pitch_onset = df_to_pitch_onset_features(df_annotation)
        f_DLNCO = pitch_onset_features_to_DLNCO(
            f_peaks=f_pitch_onset,
            feature_rate=feature_rate,
            feature_sequence_length=f_chroma_quantized.shape[1],
            visualize=visualize,
        )

        return f_chroma_quantized, f_DLNCO

    def _shift_chroma_vectors(
        self,
        f_chroma_quantized_audio: np.ndarray,
        f_chroma_quantized_annotation: np.ndarray,
        f_DLNCO_annotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        f_cens_1hz_audio = quantized_chroma_to_CENS(
            f_chroma_quantized_audio, 201, 50, self.feature_rate
        )[0]
        f_cens_1hz_annotation = quantized_chroma_to_CENS(
            f_chroma_quantized_annotation, 201, 50, self.feature_rate
        )[0]
        opt_chroma_shift = compute_optimal_chroma_shift(
            f_cens_1hz_audio, f_cens_1hz_annotation
        )
        self.logger.info(
            f"Pitch shift between the audio recording and score, determined by DTW: {opt_chroma_shift} bins"
        )
        f_chroma_quantized_annotation = shift_chroma_vectors(
            f_chroma_quantized_annotation, opt_chroma_shift
        )
        f_DLNCO_annotation = shift_chroma_vectors(f_DLNCO_annotation, opt_chroma_shift)
        return f_chroma_quantized_annotation, f_DLNCO_annotation

    def align(self, wav_file: bytes, midi_file: bytes) -> tuple[list[AlignedMidi], int, list[str]]:
        wav_buffer = BytesIO(wav_file)
        wav_buffer.seek(0)
        audio, _ = librosa.load(wav_buffer, sr=self.sr)
        audio_trim, index = librosa.effects.trim(audio, top_db=self.trim_top_db)
        start_sample, _ = index
        start_sec = float(start_sample) / float(self.sr)
        self.logger.info(f"start_sec: {start_sec}")

        df_annotation, instrument_names = self._convert_midi_to_dataframe(midi_file)
        num_instruments = len(df_annotation["instrument"].unique())
        self.logger.info(f"num_instruments: {num_instruments}")

        tuning_offset = estimate_tuning(audio_trim, self.sr)
        self.logger.info(
            f"Estimated tuning deviation for recording: {tuning_offset} cents"
        )

        f_chroma_quantized_audio, f_DLNCO_audio = self._get_features_from_audio(
            audio=audio_trim,
            tuning_offset=tuning_offset,
            Fs=self.sr,
            feature_rate=self.feature_rate,
        )

        aligned_midi_list: list[AlignedMidi] = []
        for instrument, df_inst in df_annotation.groupby("instrument"):
            self.logger.info(f"Processing instrument: {instrument}")

            df_inst = df_inst.reset_index(drop=True)
            f_chroma_quantized_annotation, f_DLNCO_annotation = (
                self._get_features_from_annotation(
                    df_annotation=df_inst, feature_rate=self.feature_rate
                )
            )
            f_chroma_quantized_annotation, f_DLNCO_annotation = (
                self._shift_chroma_vectors(
                    f_chroma_quantized_audio=f_chroma_quantized_audio,
                    f_chroma_quantized_annotation=f_chroma_quantized_annotation,
                    f_DLNCO_annotation=f_DLNCO_annotation,
                )
            )
            wp = sync_via_mrmsdtw(
                f_chroma1=f_chroma_quantized_audio,
                f_onset1=f_DLNCO_audio,
                f_chroma2=f_chroma_quantized_annotation,
                f_onset2=f_DLNCO_annotation,
                input_feature_rate=self.feature_rate,
                step_weights=self.step_weight,
                threshold_rec=self.threshold_rec,
                verbose=True,
            )
            self.logger.info(
                f"Length of warping path obtained from MrMsDTW: {wp.shape[1]}"
            )
            wp = make_path_strictly_monotonic(wp)
            self.logger.info(
                f"Length of warping path made strictly monotonic: {wp.shape[1]}"
            )

            df_annotation_warped = df_inst.copy(deep=True)
            df_annotation_warped["end"] = (
                df_annotation_warped["start"] + df_annotation_warped["duration"]
            )
            df_annotation_warped[["start", "end"]] = scipy.interpolate.interp1d(
                wp[1] / self.feature_rate,
                wp[0] / self.feature_rate,
                kind="linear",
                fill_value="extrapolate",
            )(df_inst[["start", "end"]])
            df_annotation_warped["duration"] = (
                df_annotation_warped["end"] - df_annotation_warped["start"]
            )

            tuning_multiplier = 2 ** (tuning_offset / 1200.0)
            df_annotation_warped["frequency"] = (
                440.0
                * 2 ** ((df_annotation_warped["pitch"] - 69) / 12.0)
                * tuning_multiplier
            )

            out_df = df_annotation_warped[["start", "frequency", "duration"]]
            aligned_midi = AlignedMidi(
                notes=[
                    Note(
                        start=row["start"] + start_sec,
                        frequency=row["frequency"],
                        duration=row["duration"],
                    )
                    for _, row in out_df.iterrows()
                ]
            )
            aligned_midi_list.append(aligned_midi)
        return aligned_midi_list, num_instruments, instrument_names
