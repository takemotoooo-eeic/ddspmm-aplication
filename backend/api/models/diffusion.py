import json
import os
import time
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel

from api.config import LossConfig, PreprocessConfig, TrainConfig
from api.config.model.diffusion_model import DiffusionModelConfig
from api.controllers.backend_api.openapi import models
from api.libs.logging import get_logger
from api.libs.note import AlignedMidi, Note
from api.libs.wav import preprocess_wav_file
from api.libs.const import (
    DIFFUSION_MODEL_PATH,
    DIFFUSION_CONFIG_PATH,
    DIFFUSION_STATISTICS_PATH,
    INSTRUMENT_MAPPING_PATH,
    DEFAULT_SAMPLING_RATE,
    DEFAULT_BLOCK_SIZE,
    PREPROCESS_CONFIG_PATH,
    TRAIN_CONFIG_PATH,
    URMP_LOUDNESS_SCORE_SILENCE,
    URMP_LOUDNESS_SCORE_LOUD,
)
from api.models.loss import Loss

from api.models.model import Diffuser, get_diffusion_model
from api.libs.core import normalize, split_params
from api.libs.instrument import resolve_gm_instrument_code

from .ddsp import get_ddsp_decoder


class DiffusionGenerateParams(BaseModel):
    """Diffusion生成APIのパラメータ"""
    notes: list[Note]  # 音符列
    instrument_name: str  # 楽器名（instrument_mapping.jsonのキー）
    signal_length: int  # 信号長
    num_denoising_steps: int | None = None  # デノイジング回数（省略時はモデル設定値）
    use_ddim: bool | None = None
    note_operation: Literal["add", "delete", "move", "resize"] | None = None
    operation_prev_note: Note | None = None
    operation_note: Note | None = None
    prev_features: models.DDSPGenerateParams | None = None  # 編集前の生成パラメータ


class DiffusionTrainInput(BaseModel):
    wav_file: bytes
    num_instruments: int
    instrument_names: list[str]
    midi: list[AlignedMidi]


class DiffusionModel:
    """Diffusionモデルを使用して合成パラメータを生成するクラス"""
    
    def __init__(self):
        self.logger = get_logger()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"device: {self.device}")
        self.ddsp_decoder: torch.nn.Module | None = None

        self._load_model()
        self.logger.info("Finished loading diffusion model")
    
    def _load_model(self):
        """Diffusionモデルと設定を読み込む"""
        # 設定ファイルの読み込み（簡易版、実際にはyamlをパースする必要がある）
        # ここでは、設定ファイルが存在することを確認するだけ
        if not os.path.exists(DIFFUSION_CONFIG_PATH):
            raise FileNotFoundError(f"Config file not found: {DIFFUSION_CONFIG_PATH}")
        if not os.path.exists(DIFFUSION_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {DIFFUSION_MODEL_PATH}")
        if not os.path.exists(DIFFUSION_STATISTICS_PATH):
            raise FileNotFoundError(f"Statistics file not found: {DIFFUSION_STATISTICS_PATH}")
        if not os.path.exists(INSTRUMENT_MAPPING_PATH):
            raise FileNotFoundError(f"Instrument mapping file not found: {INSTRUMENT_MAPPING_PATH}")
        
        # 統計情報を読み込む
        with open(DIFFUSION_STATISTICS_PATH, "r") as f:
            self.statistics = json.load(f)
        
        # instrument_mappingを読み込む
        with open(INSTRUMENT_MAPPING_PATH, "r") as f:
            self.instrument_mapping = json.load(f)
        
        # 統計情報を設定
        self.f0_original_mean = self.statistics["f0_original"]["mean"]
        self.f0_original_std = self.statistics["f0_original"]["std"]
        self.f0_score_mean = self.f0_original_mean
        self.f0_score_std = self.f0_original_std
        self.f0_diff_mean = self.statistics["f0_diff"]["mean"]
        self.f0_diff_std = self.statistics["f0_diff"]["std"]
        self.loudness_original_mean = self.statistics["loudness_original_v2"]["mean"]
        self.loudness_original_std = self.statistics["loudness_original_v2"]["std"]
        self.loudness_diff_mean = self.statistics["loudness_diff"]["mean"]
        self.loudness_diff_std = self.statistics["loudness_diff"]["std"]
        
        z_feature_mean_list = self.statistics["z_feature"]["mean"]
        z_feature_std_list = self.statistics["z_feature"]["std"]
        self.z_feature_mean = torch.tensor(z_feature_mean_list, device=self.device).reshape(1, 1, -1)
        self.z_feature_std = torch.tensor(z_feature_std_list, device=self.device).reshape(1, 1, -1)
        
        # 設定ファイルから設定を読み込む
        self.diffusion_model_config = DiffusionModelConfig.from_config_path(DIFFUSION_CONFIG_PATH)
        
        # モデルを読み込む
        state_dict = torch.load(DIFFUSION_MODEL_PATH, map_location=self.device)
        
        # state_dictから楽器数を推測
        if "instrument_emb.weight" in state_dict:
            num_instruments = state_dict["instrument_emb.weight"].shape[0]
        else:
            num_instruments = len(self.instrument_mapping)
            self.logger.warning(f"instrument_emb.weight not found, using instrument count: {num_instruments}")
        
        
        self.diffusion_model = get_diffusion_model(
            time_embed_dim=self.diffusion_model_config.time_embed_dim,
            num_instruments=len(self.instrument_mapping),
            film_cond_dim=self.diffusion_model_config.film_cond_dim,
            cfg_dropout_prob=self.diffusion_model_config.cfg_dropout_prob,
            f0_score_scale=self.diffusion_model_config.f0_score_scale,
            # UNet parameters
            down1_out_ch=self.diffusion_model_config.down1_out_ch,
            down2_out_ch=self.diffusion_model_config.down2_out_ch,
            down3_out_ch=self.diffusion_model_config.down3_out_ch,
            bot1_out_ch=self.diffusion_model_config.bot1_out_ch,
        ).to(self.device)
        
        # 重みを読み込む
        missing_keys, unexpected_keys = self.diffusion_model.load_state_dict(state_dict, strict=True)
        if missing_keys:
            self.logger.warning(f"Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            self.logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")
        self.logger.info(f"Loaded diffusion model from {DIFFUSION_MODEL_PATH}")
        
        # Diffuserを作成（設定ファイルから読み込んだ値を使用）
        self.diffuser = Diffuser(
            num_timesteps=self.diffusion_model_config.num_timesteps,
            beta_start=self.diffusion_model_config.beta_start,
            beta_end=self.diffusion_model_config.beta_end,
            device=str(self.device),
        )
        
        self.cfg_guidance_scale = 2.0

    def _notes_to_score_values(
        self,
        notes: list[Note],
        total_length: int,
        sampling_rate: int,
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        音符列からf0_scoreとloudness_scoreを生成
        DDSPMMの_si_ddspmm_diffusion_init.pyのinitialize_tensorsを参考
        """
        sorted_notes = sorted(notes, key=lambda n: n.start)
        mean_frequency = np.mean([note.frequency for note in sorted_notes])
        f0_scores = np.full(total_length, mean_frequency)
        loudness_scores = np.full(total_length, URMP_LOUDNESS_SCORE_SILENCE)
        
        for i, note in enumerate(sorted_notes):
            start_sample = int(note.start * sampling_rate)
            end_sample = int((note.start + note.duration) * sampling_rate)

            if start_sample < total_length:
                end_sample = min(end_sample, total_length)
                
                if i == 0:
                    f0_scores[:start_sample] = note.frequency
                else:
                    f0_scores[last_end_sample:start_sample] = last_frequency
                
                f0_scores[start_sample:end_sample] = note.frequency
                loudness_scores[start_sample:end_sample] = URMP_LOUDNESS_SCORE_LOUD
                
                last_frequency = note.frequency
                last_end_sample = end_sample
            
            if len(sorted_notes) > 0 and last_end_sample < total_length:
                f0_scores[last_end_sample:] = last_frequency
        
        f0_scores = f0_scores[::block_size]
        loudness_scores = loudness_scores[::block_size]
        return f0_scores, loudness_scores

    def _build_training_tensors(
        self,
        train_input: DiffusionTrainInput,
        preprocess_config: PreprocessConfig,
    ) -> dict:
        signal_mix = preprocess_wav_file(
            train_input.wav_file, preprocess_config, self.device
        )
        padded_length = (
            preprocess_config.signal_length
            - len(signal_mix) % preprocess_config.signal_length
        ) % preprocess_config.signal_length
        mix_signal = torch.nn.functional.pad(signal_mix, (0, padded_length))
        total_length = len(mix_signal)
        segment_num = total_length // preprocess_config.signal_length

        f0_scores_list: list[np.ndarray] = []
        loudness_scores_list: list[np.ndarray] = []
        instrument_ids: list[int] = []

        for aligned_midi, instrument_name in zip(train_input.midi, train_input.instrument_names):
            mapping_key = resolve_gm_instrument_code(instrument_name)
            if mapping_key not in self.instrument_mapping:
                raise ValueError(
                    f"Instrument {instrument_name} (mapping key: {mapping_key}) not found in mapping. "
                    f"Available: {list(self.instrument_mapping.keys())}"
                )
            instrument_ids.append(self.instrument_mapping[mapping_key])
            f0_scores, loudness_scores = self._notes_to_score_values(
                notes=aligned_midi.notes,
                total_length=total_length,
                sampling_rate=preprocess_config.sampling_rate,
                block_size=preprocess_config.block_size
            )
            f0_scores_list.append(f0_scores.reshape(segment_num, -1))
            loudness_scores_list.append(loudness_scores.reshape(segment_num, -1))

        f0_scores_tensor = torch.stack(
            [torch.from_numpy(f0).to(self.device) for f0 in f0_scores_list]
        ).float()
        loudness_scores_tensor = torch.stack(
            [torch.from_numpy(l).to(self.device) for l in loudness_scores_list]
        ).float()

        segment_num_dim = f0_scores_tensor.shape[1]
        L_seg = f0_scores_tensor.shape[2]
        num_instruments = train_input.num_instruments

        all_f0_scores = f0_scores_tensor.reshape(-1, L_seg)
        all_loudness_scores = loudness_scores_tensor.reshape(-1, L_seg)
        all_instrument_ids = torch.tensor(instrument_ids, device=self.device).repeat_interleave(
            segment_num_dim
        )

        return {
            "observed_mixture": mix_signal.unsqueeze(0),
            "all_f0_scores": all_f0_scores,
            "all_loudness_scores": all_loudness_scores,
            "all_instrument_ids": all_instrument_ids,
            "all_f0_scores_normalized": normalize(
                all_f0_scores, self.f0_score_mean, self.f0_score_std
            ).float(),
            "segment_num": segment_num_dim,
            "L_seg": L_seg,
            "num_instruments": num_instruments,
        }

    def train(self, train_input: DiffusionTrainInput) -> models.Features:
        """WAV+MIDIをアラインしたスコアから、DDSPガイダンス付きDiffusionでパラメータを生成する。"""
        train_config = TrainConfig.from_config_path(TRAIN_CONFIG_PATH)
        loss_config = LossConfig.from_config_path(TRAIN_CONFIG_PATH)
        preprocess_config = PreprocessConfig.from_config_path(PREPROCESS_CONFIG_PATH)

        tensors = self._build_training_tensors(train_input, preprocess_config)
        segment_num = tensors["segment_num"]
        L_seg = tensors["L_seg"]
        num_instruments = tensors["num_instruments"]
        batch_size = num_instruments * segment_num

        self.logger.info(
            f"Diffusion train: instruments={num_instruments}, segments={segment_num}, L_seg={L_seg}"
        )

        loss_fn = Loss(self.device, loss_config)
        self.diffusion_model.eval()

        if train_config.enable_guidance:
            if self.ddsp_decoder is None:
                self.ddsp_decoder = get_ddsp_decoder()
            self.logger.info("Sampling with DDSP guidance")
            all_params_sampled = self.diffuser.sample_with_guidance(
                model=self.diffusion_model,
                shape=(batch_size, L_seg, 18),
                instrument_ids=tensors["all_instrument_ids"],
                f0_score=tensors["all_f0_scores_normalized"],
                ddsp_decoder=self.ddsp_decoder,
                loss_fn=loss_fn,
                loss_config=loss_config,
                split_params_fn=split_params,
                observed_mixture=tensors["observed_mixture"],
                f0_diff_mean=self.f0_diff_mean,
                f0_diff_std=self.f0_diff_std,
                f0_original_mean=self.f0_original_mean,
                f0_original_std=self.f0_original_std,
                z_feature_mean=self.z_feature_mean,
                z_feature_std=self.z_feature_std,
                loudness_diff_mean=self.loudness_diff_mean,
                loudness_diff_std=self.loudness_diff_std,
                loudness_original_mean=self.loudness_original_mean,
                loudness_original_std=self.loudness_original_std,
                loudness_score=tensors["all_loudness_scores"],
                guidance_scale_start=train_config.guidance_scale_start,
                guidance_scale_end=train_config.guidance_scale_end,
                instrument_names=train_input.instrument_names,
                segment_num=segment_num,
                num_instruments=num_instruments,
                sampling_rate=preprocess_config.sampling_rate,
                guiding_params=train_config.guiding_params,
            )
        else:
            self.logger.info("Sampling without guidance")
            with torch.no_grad():
                all_params_sampled = self.diffuser.sample(
                    model=self.diffusion_model,
                    shape=(batch_size, L_seg, 18),
                    instrument_ids=tensors["all_instrument_ids"],
                    f0_score=tensors["all_f0_scores_normalized"],
                )

        all_f0s, _, _, all_loudness_diff, all_z_features = split_params(
            aggregated=all_params_sampled,
            f0_original_mean=self.f0_original_mean,
            f0_original_std=self.f0_original_std,
            f0_diff_mean=self.f0_diff_mean,
            f0_diff_std=self.f0_diff_std,
            loudness_original_mean=self.loudness_original_mean,
            loudness_original_std=self.loudness_original_std,
            loudness_diff_mean=self.loudness_diff_mean,
            loudness_diff_std=self.loudness_diff_std,
            z_feature_mean=self.z_feature_mean,
            z_feature_std=self.z_feature_std,
            loudness_score=tensors["all_loudness_scores"],
            f0_score=tensors["all_f0_scores_normalized"],
        )
        all_loudness_scores = tensors["all_loudness_scores"]

        features: list[models.Feature] = []
        for i in range(num_instruments):
            instrument_pitches = []
            instrument_loudnesses = []
            instrument_z_features = []
            for seg_idx in range(segment_num):
                batch_idx = i * segment_num + seg_idx
                instrument_pitches.append(all_f0s[batch_idx])
                instrument_loudnesses.append(
                    all_loudness_diff[batch_idx] + all_loudness_scores[batch_idx]
                )
                instrument_z_features.append(all_z_features[batch_idx])

            pitch_cat = torch.cat(instrument_pitches, dim=0)
            loudness_cat = torch.cat(instrument_loudnesses, dim=0)
            z_feature_cat = torch.cat(instrument_z_features, dim=0)

            features.append(
                models.Feature(
                    instrument_name=train_input.instrument_names[i],
                    pitch=pitch_cat.detach().cpu().numpy().tolist(),
                    loudness=loudness_cat.detach().cpu().numpy().tolist(),
                    z_feature=z_feature_cat.detach().cpu().numpy().tolist(),
                    notes=[
                        models.Note(
                            start=note.start,
                            frequency=note.frequency,
                            duration=note.duration,
                        )
                        for note in train_input.midi[i].notes
                    ],
                )
            )

        return models.Features(features=features)

    def generate(
        self,
        params: DiffusionGenerateParams,
    ) -> dict:
        """
        音符列と楽器IDから合成パラメータを生成
        """
        mapping_key = resolve_gm_instrument_code(params.instrument_name)
        if mapping_key not in self.instrument_mapping:
            raise ValueError(
                f"Instrument {params.instrument_name} (mapping key: {mapping_key}) not found in mapping. "
                f"Available: {list(self.instrument_mapping.keys())}"
            )

        instrument_id = self.instrument_mapping[mapping_key]
        self.logger.info(
            f"generate: {params.instrument_name} -> mapping {mapping_key} -> id {instrument_id}"
        )
        self._log_generate_target_notes(params)
        
        sampling_rate = DEFAULT_SAMPLING_RATE
        block_size = DEFAULT_BLOCK_SIZE
        
        signal_length = params.signal_length
        partial_ranges = self._calculate_partial_generate_ranges(
            params=params,
            block_size=block_size,
        )

        full_f0_scores, full_loudness_scores = self._notes_to_score_values(
            notes=params.notes,
            total_length=signal_length,
            sampling_rate=sampling_rate,
            block_size=block_size,
        )
        if partial_ranges is None:
            f0_scores_tensor = torch.from_numpy(full_f0_scores).to(self.device).float().unsqueeze(0)
            loudness_scores_tensor = torch.from_numpy(full_loudness_scores).to(self.device).float().unsqueeze(0)
            L_seg = len(full_f0_scores)
            self.logger.info(
                f"Diffusion generate region: full feature[0:{L_seg}] "
                f"(0.000s - {signal_length / sampling_rate:.3f}s)"
            )
        else:
            f0_score_segments = []
            loudness_score_segments = []
            for batch_idx, (feature_start, feature_end) in enumerate(partial_ranges):
                start_sec = feature_start * block_size / sampling_rate
                end_sec = feature_end * block_size / sampling_rate
                self.logger.info(
                    f"Diffusion generate region batch={batch_idx}: feature[{feature_start}:{feature_end}] "
                    f"({start_sec:.3f}s - {end_sec:.3f}s)"
                )
                f0_score_segments.append(full_f0_scores[feature_start:feature_end])
                loudness_score_segments.append(full_loudness_scores[feature_start:feature_end])

            f0_scores_tensor = torch.from_numpy(np.stack(f0_score_segments)).to(self.device).float()
            loudness_scores_tensor = torch.from_numpy(np.stack(loudness_score_segments)).to(self.device).float()
            L_seg = partial_ranges[0][1] - partial_ranges[0][0]
            changed_feature_ranges = self._calculate_changed_feature_ranges(
                params=params,
                block_size=block_size,
                feature_length=len(params.prev_features.pitch) if params.prev_features else 0,
            )
            inpaint_mask = self._build_inpaint_mask(
                partial_ranges=partial_ranges,
                changed_feature_ranges=changed_feature_ranges,
                segment_length=L_seg,
            )
            previous_feature_segments = self._build_previous_feature_segments(
                prev_features=params.prev_features,
                partial_ranges=partial_ranges,
                f0_scores=f0_scores_tensor,
                loudness_scores=loudness_scores_tensor,
            )

        batch_size = f0_scores_tensor.shape[0]
        instrument_ids_tensor = torch.full(
            (batch_size,), instrument_id, device=self.device, dtype=torch.long
        )
        
        # f0_scoreを正規化
        f0_scores_normalized = normalize(f0_scores_tensor, self.f0_score_mean, self.f0_score_std).float()
        
        num_denoising_steps = params.num_denoising_steps
        if num_denoising_steps is not None:
            max_steps = self.diffusion_model_config.num_timesteps
            if num_denoising_steps < 1 or num_denoising_steps > max_steps:
                raise ValueError(
                    f"num_denoising_steps must be between 1 and {max_steps}, "
                    f"got {num_denoising_steps}"
                )


        # Diffusion samplingでパラメータを生成
        self.logger.info(
            f"Sampling parameters from diffusion model "
            f"(num_denoising_steps={num_denoising_steps or self.diffusion_model_config.num_timesteps}, "
            f"sampler={'DDIM' if params.use_ddim else 'DDPM'})"
        )
        self.diffusion_model.eval()
        
        start_time = time.time()
        with torch.no_grad():
            if partial_ranges is not None and params.prev_features is not None:
                params_sampled = self.diffuser.inpaint(
                    model=self.diffusion_model,
                    shape=(batch_size, L_seg, 18),
                    instrument_ids=instrument_ids_tensor,
                    f0_score=f0_scores_normalized,
                    mask=inpaint_mask,
                    previous_features=previous_feature_segments,
                    num_denoising_steps=num_denoising_steps,
                    use_ddim=params.use_ddim,
                )
            else:
                params_sampled = self.diffuser.sample(
                    model=self.diffusion_model,
                    shape=(batch_size, L_seg, 18),
                    instrument_ids=instrument_ids_tensor,
                    f0_score=f0_scores_normalized,
                    num_denoising_steps=num_denoising_steps,
                    use_ddim=params.use_ddim,
                )  # (batch_size, L_seg, 18)
        end_time = time.time()
        self.logger.info(f"Time taken: {end_time - start_time} seconds")
        all_f0s, _, all_loudness_normalized, _, all_z_features = split_params(
            aggregated=params_sampled,
            f0_original_mean=self.f0_original_mean,
            f0_original_std=self.f0_original_std,
            f0_diff_mean=self.f0_diff_mean,
            f0_diff_std=self.f0_diff_std,
            loudness_original_mean=self.loudness_original_mean,
            loudness_original_std=self.loudness_original_std,
            loudness_diff_mean=self.loudness_diff_mean,
            loudness_diff_std=self.loudness_diff_std,
            z_feature_mean=self.z_feature_mean,
            z_feature_std=self.z_feature_std,
            loudness_score=loudness_scores_tensor,
            f0_score=f0_scores_normalized,
        )
        all_loudness: torch.Tensor = all_loudness_normalized * self.loudness_original_std + self.loudness_original_mean

        all_f0s_np = all_f0s.detach().cpu().numpy()  # (batch_size, L_seg)
        all_loudnesses_np = all_loudness.detach().cpu().numpy()  # (batch_size, L_seg)
        all_z_features_np = all_z_features.detach().cpu().numpy()  # (batch_size, L_seg, 16)

        if partial_ranges is not None and params.prev_features is not None:
            return self._merge_partial_features(
                prev_features=params.prev_features,
                partial_ranges=partial_ranges,
                pitch=all_f0s_np,
                loudness=all_loudnesses_np,
                z_feature=all_z_features_np,
            )

        return {
            "pitch": all_f0s_np.squeeze(0).tolist(),
            "loudness": all_loudnesses_np.squeeze(0).tolist(),
            "z_feature": all_z_features_np.squeeze(0).tolist(),
        }

    def _log_generate_target_notes(self, params: DiffusionGenerateParams) -> None:
        if params.note_operation is None:
            self.logger.info(
                f"Diffusion generate target: full regenerate notes={len(params.notes)}"
            )
            return

        self.logger.info(
            "Diffusion generate target: "
            f"operation={params.note_operation}, "
            f"prev_note={self._format_note_for_log(params.operation_prev_note)}, "
            f"note={self._format_note_for_log(params.operation_note)}"
        )

    def _format_note_for_log(self, note: Note | None) -> str:
        if note is None:
            return "None"
        return (
            "{"
            f"start={note.start:.3f}, "
            f"duration={note.duration:.3f}, "
            f"end={note.start + note.duration:.3f}, "
            f"frequency={note.frequency:.2f}"
            "}"
        )

    def _calculate_partial_generate_ranges(
        self,
        params: DiffusionGenerateParams,
        block_size: int,
    ) -> list[tuple[int, int]] | None:
        prev_features = params.prev_features
        if prev_features is None:
            return None
        feature_length = len(prev_features.pitch)
        if (
            feature_length == 0
            or len(prev_features.loudness) != feature_length
            or len(prev_features.z_feature) != feature_length
        ):
            return None

        signal_feature_length = int(np.ceil(params.signal_length / block_size))
        feature_length = min(feature_length, signal_feature_length)
        segment_feature_length = self._segment_feature_length(block_size)
        window_length = min(segment_feature_length, feature_length)

        changed_feature_ranges = self._calculate_changed_feature_ranges(
            params=params,
            feature_length=feature_length,
            block_size=block_size,
        )
        if not changed_feature_ranges:
            return None

        window_starts: set[int] = set()
        if params.note_operation == "move" and len(changed_feature_ranges) == 2:
            combined_start = min(start for start, _ in changed_feature_ranges)
            combined_end = max(end for _, end in changed_feature_ranges)
            if combined_end - combined_start <= window_length:
                for window_start in self._centered_window_starts(
                    feature_start=combined_start,
                    feature_end=combined_end,
                    feature_length=feature_length,
                    window_length=window_length,
                ):
                    window_starts.add(window_start)

        if not window_starts:
            for feature_start, feature_end in changed_feature_ranges:
                for window_start in self._centered_window_starts(
                    feature_start=feature_start,
                    feature_end=feature_end,
                    feature_length=feature_length,
                    window_length=window_length,
                ):
                    window_starts.add(window_start)

        partial_ranges = [
            (start, start + window_length)
            for start in sorted(window_starts)
            if start < feature_length
        ]
        if not partial_ranges:
            return None
        return partial_ranges

    def _calculate_changed_feature_ranges(
        self,
        params: DiffusionGenerateParams,
        feature_length: int,
        block_size: int,
    ) -> list[tuple[int, int]]:
        if feature_length <= 0:
            return []
        changed_ranges = self._calculate_changed_note_ranges(
            operation=params.note_operation,
            prev_note=params.operation_prev_note,
            note=params.operation_note,
        )
        if not changed_ranges:
            return []
        signal_feature_length = int(np.ceil(params.signal_length / block_size))
        return self._changed_ranges_to_feature_ranges(
            changed_ranges=changed_ranges,
            feature_length=min(feature_length, signal_feature_length),
            block_size=block_size,
        )

    def _build_inpaint_mask(
        self,
        partial_ranges: list[tuple[int, int]],
        changed_feature_ranges: list[tuple[int, int]],
        segment_length: int,
    ) -> torch.Tensor:
        mask = torch.zeros(
            (len(partial_ranges), segment_length), device=self.device, dtype=torch.float32
        )
        for batch_idx, (window_start, window_end) in enumerate(partial_ranges):
            for changed_start, changed_end in changed_feature_ranges:
                overlap_start = max(window_start, changed_start)
                overlap_end = min(window_end, changed_end)
                if overlap_end <= overlap_start:
                    continue
                mask[
                    batch_idx,
                    overlap_start - window_start : overlap_end - window_start,
                ] = 1.0
        return mask

    def _build_previous_feature_segments(
        self,
        prev_features: models.DDSPGenerateParams,
        partial_ranges: list[tuple[int, int]],
        f0_scores: torch.Tensor,
        loudness_scores: torch.Tensor,
    ) -> torch.Tensor:
        pitch = torch.tensor(prev_features.pitch, device=self.device).float()
        loudness = torch.tensor(prev_features.loudness, device=self.device).float()
        z_feature = torch.tensor(prev_features.z_feature, device=self.device).float()

        pitch_segments = torch.stack(
            [pitch[start:end] for start, end in partial_ranges]
        )
        loudness_segments = torch.stack(
            [loudness[start:end] for start, end in partial_ranges]
        )
        z_feature_segments = torch.stack(
            [z_feature[start:end] for start, end in partial_ranges]
        )

        f0_diff = pitch_segments - f0_scores
        loudness_diff = loudness_segments - loudness_scores
        return torch.cat(
            [
                normalize(f0_diff, self.f0_diff_mean, self.f0_diff_std).unsqueeze(-1),
                normalize(
                    loudness_diff, self.loudness_diff_mean, self.loudness_diff_std
                ).unsqueeze(-1),
                normalize(z_feature_segments, self.z_feature_mean, self.z_feature_std),
            ],
            dim=-1,
        )

    def _changed_ranges_to_feature_ranges(
        self,
        changed_ranges: list[tuple[float, float]],
        feature_length: int,
        block_size: int,
    ) -> list[tuple[int, int]]:
        feature_ranges: list[tuple[int, int]] = []
        for changed_start, changed_end in changed_ranges:
            feature_start = max(
                0,
                int(np.floor(changed_start * DEFAULT_SAMPLING_RATE / block_size)),
            )
            feature_end = min(
                feature_length,
                int(np.ceil(changed_end * DEFAULT_SAMPLING_RATE / block_size)),
            )
            if feature_end <= feature_start:
                feature_end = min(feature_length, feature_start + 1)
            if feature_end > feature_start:
                feature_ranges.append((feature_start, feature_end))
        return feature_ranges

    def _centered_window_starts(
        self,
        feature_start: int,
        feature_end: int,
        feature_length: int,
        window_length: int,
    ) -> list[int]:
        if window_length <= 0:
            return []
        max_start = max(0, feature_length - window_length)
        changed_length = feature_end - feature_start
        if changed_length <= window_length:
            changed_center = (feature_start + feature_end) / 2
            return [int(np.clip(round(changed_center - window_length / 2), 0, max_start))]

        starts: list[int] = []
        chunk_start = feature_start
        while chunk_start < feature_end:
            chunk_end = min(feature_end, chunk_start + window_length)
            chunk_center = (chunk_start + chunk_end) / 2
            starts.append(
                int(np.clip(round(chunk_center - window_length / 2), 0, max_start))
            )
            chunk_start = chunk_end
        return starts

    def _segment_feature_length(self, block_size: int) -> int:
        preprocess_config = PreprocessConfig.from_config_path(PREPROCESS_CONFIG_PATH)
        return max(1, int(np.ceil(preprocess_config.signal_length / block_size)))

    def _calculate_changed_note_ranges(
        self,
        operation: Literal["add", "delete", "move", "resize"] | None,
        prev_note: Note | None,
        note: Note | None,
    ) -> list[tuple[float, float]]:
        if operation is None:
            return []

        if operation == "add" and note is not None:
            return [(note.start, note.start + note.duration)]

        if operation == "delete" and prev_note is not None:
            return [(prev_note.start, prev_note.start + prev_note.duration)]

        if operation == "move" and prev_note is not None and note is not None:
            return [
                (prev_note.start, prev_note.start + prev_note.duration),
                (note.start, note.start + note.duration),
            ]

        if operation == "resize" and prev_note is not None and note is not None:
            return [self._resize_changed_range(prev_note, note)]

        return []

    def _resize_changed_range(self, prev_note: Note, note: Note) -> tuple[float, float]:
        prev_start = prev_note.start
        prev_end = prev_note.start + prev_note.duration
        note_start = note.start
        note_end = note.start + note.duration
        if self._contains_interval(prev_start, prev_end, note_start, note_end):
            return prev_start, prev_end
        return note_start, note_end

    def _contains_interval(
        self,
        outer_start: float,
        outer_end: float,
        inner_start: float,
        inner_end: float,
    ) -> bool:
        return outer_start <= inner_start and inner_end <= outer_end

    def _merge_partial_features(
        self,
        prev_features: models.DDSPGenerateParams,
        partial_ranges: list[tuple[int, int]],
        pitch: np.ndarray,
        loudness: np.ndarray,
        z_feature: np.ndarray,
    ) -> dict:
        merged_pitch = list(prev_features.pitch)
        merged_loudness = list(prev_features.loudness)
        merged_z_feature = [list(row) for row in prev_features.z_feature]
        for batch_idx, (feature_start, feature_end) in enumerate(partial_ranges):
            replace_len = min(feature_end - feature_start, pitch.shape[1])
            end = feature_start + replace_len
            merged_pitch[feature_start:end] = pitch[batch_idx, :replace_len].tolist()
            merged_loudness[feature_start:end] = loudness[batch_idx, :replace_len].tolist()
            merged_z_feature[feature_start:end] = z_feature[batch_idx, :replace_len].tolist()
        return {
            "pitch": merged_pitch,
            "loudness": merged_loudness,
            "z_feature": merged_z_feature,
        }


