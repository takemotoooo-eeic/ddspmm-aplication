import json
import os
import time
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
        signal_mix, _, _ = preprocess_wav_file(
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

        train_config.enable_guidance = False
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
        
        sampling_rate = DEFAULT_SAMPLING_RATE
        block_size = DEFAULT_BLOCK_SIZE
        
        # 信号長を使用
        signal_length = params.signal_length
        
        f0_scores, loudness_scores = self._notes_to_score_values(
            notes=params.notes,
            total_length=signal_length,
            sampling_rate=sampling_rate,
            block_size=block_size,
        )
        
        L_seg = len(f0_scores)  # セグメント長
        
        # テンソルに変換
        f0_scores_tensor = torch.from_numpy(f0_scores).to(self.device).float().unsqueeze(0)  # (1, L_seg)
        loudness_scores_tensor = torch.from_numpy(loudness_scores).to(self.device).float().unsqueeze(0)  # (1, L_seg)
        instrument_ids_tensor = torch.tensor([instrument_id], device=self.device)  # (1,)
        
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
            f"(num_denoising_steps={num_denoising_steps or self.diffusion_model_config.num_timesteps})"
        )
        self.diffusion_model.eval()
        
        start_time = time.time()
        with torch.no_grad():
            params_sampled = self.diffuser.sample(
                model=self.diffusion_model,
                shape=(1, L_seg, 18),
                instrument_ids=instrument_ids_tensor,
                f0_score=f0_scores_normalized,
                num_denoising_steps=num_denoising_steps,
            )  # (1, L_seg, 18)
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
        self.logger.info(f"all_loudness: {all_loudness}")

        all_f0s_np = all_f0s.squeeze(0).detach().cpu().numpy()  # (L_seg,)
        all_loudnesses_np = all_loudness.squeeze(0).detach().cpu().numpy()  # (L_seg,)
        all_z_features_np = all_z_features.squeeze(0).detach().cpu().numpy()  # (L_seg, 16)
        
        return {
            "pitch": all_f0s_np.tolist(),
            "loudness": all_loudnesses_np.tolist(),
            "z_feature": all_z_features_np.tolist(),
        }

