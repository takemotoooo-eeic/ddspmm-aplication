"""
Diffusionモデルを使用して合成パラメータを生成するクラス
DDSPMMの_si_ddspmm_diffusion_init.pyを参考に実装
"""
import json
import os
from typing import Optional

import numpy as np
import torch
from pydantic import BaseModel

from api.config.model.diffusion import DiffusionModelConfig
from api.libs.logging import get_logger
from api.models.midi_aligner.midi_aligner import Note

# DDSPMMのコードをインポート（パスが設定されている場合）
# 実際の環境に応じてインポートパスを調整してください
    
from api.models.diffusion.model import Diffuser, get_diffusion_model
from api.models.diffusion.core import normalize, split_params
from api.models.diffusion.pitch import hz_to_cent


DIFFUSION_MODEL_DIR = "api/models/diffusion/model"
DIFFUSION_MODEL_PATH = os.path.join(DIFFUSION_MODEL_DIR, "models", "model_epoch_003000.pth")
DIFFUSION_CONFIG_PATH = os.path.join(DIFFUSION_MODEL_DIR, "hydra_config.yaml")
DIFFUSION_STATISTICS_PATH = os.path.join(DIFFUSION_MODEL_DIR, "statistics", "diffusion_statistics.json")
INSTRUMENT_MAPPING_PATH = os.path.join(DIFFUSION_MODEL_DIR, "instrument_mapping.json")

# デフォルト値（preprocess.config.yamlから）
DEFAULT_SAMPLING_RATE = 16000
DEFAULT_BLOCK_SIZE = 512

# 定数（DDSPMMのコードから）
URMP_LOUDNESS_SCORE_SILENCE = -65.0
URMP_LOUDNESS_SCORE_LOUD = -45.0


class DiffusionGenerateParams(BaseModel):
    """Diffusion生成APIのパラメータ"""
    notes: list[Note]  # 音符列
    instrument_name: str  # 楽器名（instrument_mapping.jsonのキー）
    signal_length: int  # 信号長


class DiffusionModel:
    """Diffusionモデルを使用して合成パラメータを生成するクラス"""
    
    def __init__(self):
        self.logger = get_logger()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"device: {self.device}")
        
        # モデルを読み込む
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
        self.f0_mean = self.statistics["f0"]["mean"]
        self.f0_std = self.statistics["f0"]["std"]
        self.f0_diff_mean = self.f0_mean
        self.f0_diff_std = self.f0_std
        self.f0_score_mean = self.statistics["original_f0"]["mean"]
        self.f0_score_std = self.statistics["original_f0"]["std"]
        self.loudness_mean = self.statistics["loudness"]["mean"]
        self.loudness_std = self.statistics["loudness"]["std"]
        self.loudness_diff_mean = self.statistics.get("loudness_diff", {}).get("mean")
        self.loudness_diff_std = self.statistics.get("loudness_diff", {}).get("std")
        
        note_mask_stats = self.statistics.get("note_mask", {})
        self.note_mask_mean = note_mask_stats.get("mean", 0.5)
        self.note_mask_std = note_mask_stats.get("std", 0.5)
        
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
        
        # モデルの作成（設定ファイルから読み込んだ値を使用）
        model_kwargs = {
            "model_type": self.diffusion_model_config.model_type,
            "in_dim": self.diffusion_model_config.in_dim,
            "time_embed_dim": self.diffusion_model_config.time_embed_dim,
            "num_instruments": num_instruments,
            "film_cond_dim": self.diffusion_model_config.film_cond_dim,
            "use_film": self.diffusion_model_config.use_film,
            "use_cross_attention": self.diffusion_model_config.use_cross_attention,
            "cfg_dropout_prob": self.diffusion_model_config.cfg_dropout_prob,
            "f0_score_scale": self.diffusion_model_config.f0_score_scale,
            "use_note_mask": self.diffusion_model_config.use_note_mask,
            "note_mask_scale": self.diffusion_model_config.note_mask_scale,
            "down1_out_ch": self.diffusion_model_config.down1_out_ch,
            "down2_out_ch": self.diffusion_model_config.down2_out_ch,
            "down3_out_ch": self.diffusion_model_config.down3_out_ch,
            "bot1_out_ch": self.diffusion_model_config.bot1_out_ch,
        }
        
        # DiTパラメータは設定がある場合のみ追加
        if hasattr(self.diffusion_model_config, "dit_hidden_dim"):
            model_kwargs["dit_hidden_dim"] = self.diffusion_model_config.dit_hidden_dim
            model_kwargs["dit_num_layers"] = self.diffusion_model_config.dit_num_layers
            model_kwargs["dit_num_heads"] = self.diffusion_model_config.dit_num_heads
            model_kwargs["dit_mlp_ratio"] = self.diffusion_model_config.dit_mlp_ratio
        
        # use_diffusersは設定ファイルから読み込む
        model_kwargs["use_diffusers"] = self.diffusion_model_config.use_diffusers
        
        self.diffusion_model = get_diffusion_model(**model_kwargs).to(self.device)
        
        # 重みを読み込む
        missing_keys, unexpected_keys = self.diffusion_model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            self.logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            self.logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        # Diffuserを作成（設定ファイルから読み込んだ値を使用）
        self.diffuser = Diffuser(
            num_timesteps=self.diffusion_model_config.num_timesteps,
            beta_start=self.diffusion_model_config.beta_start,
            beta_end=self.diffusion_model_config.beta_end,
            device=str(self.device),
        )
        
        self.cfg_guidance_scale = 2.0
        self.use_f0_diff = True
        self.use_loudness_diff = True
    
    def _notes_to_f0_score_and_mask(
        self,
        notes: list[Note],
        total_length: int,
        sampling_rate: int,
        block_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        音符列からf0_scoreとnote_maskを生成
        DDSPMMの_si_ddspmm_diffusion_init.pyのinitialize_tensorsを参考
        """
        sorted_notes = sorted(notes, key=lambda n: n.start)
        
        f0_scores = np.full(total_length, self.f0_score_mean)
        loudness_scores = np.full(total_length, URMP_LOUDNESS_SCORE_SILENCE)
        note_masks = np.zeros(total_length)
        
        for i, note in enumerate(sorted_notes):
            start_sample = int(note.start * sampling_rate)
            end_sample = int((note.start + note.duration) * sampling_rate)
            
            start_sample = max(start_sample, 0)
            end_sample = min(end_sample, total_length)
            
            next_note = sorted_notes[i + 1] if i + 1 < len(sorted_notes) else None
            if not next_note:
                f0_scores[start_sample:end_sample] = hz_to_cent(note.frequency)
                loudness_scores[start_sample:end_sample] = URMP_LOUDNESS_SCORE_LOUD
                note_masks[start_sample:end_sample] = 1.0
                continue
            
            next_start_sample = int(next_note.start * sampling_rate)
            if next_start_sample - end_sample < 512:  # frame_for_note_correction（デフォルト値）
                f0_scores[start_sample:next_start_sample] = hz_to_cent(note.frequency)
                loudness_scores[start_sample:next_start_sample] = URMP_LOUDNESS_SCORE_LOUD
                note_masks[start_sample:next_start_sample] = 1.0
                continue
            
            f0_scores[start_sample:end_sample] = hz_to_cent(note.frequency)
            loudness_scores[start_sample:end_sample] = URMP_LOUDNESS_SCORE_LOUD
            note_masks[start_sample:end_sample] = 1.0
        
        # block_sizeで間引く
        f0_scores = f0_scores[::block_size]
        loudness_scores = loudness_scores[::block_size]
        note_masks = note_masks[::block_size]
        
        return f0_scores, loudness_scores, note_masks
    
    def generate(
        self,
        params: DiffusionGenerateParams,
    ) -> dict:
        """
        音符列と楽器IDから合成パラメータを生成
        """
        if params.instrument_name not in self.instrument_mapping:
            raise ValueError(
                f"Instrument {params.instrument_name} not found in mapping. "
                f"Available: {list(self.instrument_mapping.keys())}"
            )
        
        instrument_id = self.instrument_mapping[params.instrument_name]
        print(params.instrument_name, instrument_id)
        
        sampling_rate = DEFAULT_SAMPLING_RATE
        block_size = DEFAULT_BLOCK_SIZE
        
        # 信号長を使用
        signal_length = params.signal_length
        
        # f0_scoreとnote_maskを生成
        f0_scores, loudness_scores, note_masks = self._notes_to_f0_score_and_mask(
            notes=params.notes,
            total_length=signal_length,
            sampling_rate=sampling_rate,
            block_size=block_size,
        )
        
        L_seg = len(f0_scores)  # セグメント長
        
        # テンソルに変換
        f0_scores_tensor = torch.from_numpy(f0_scores).to(self.device).float().unsqueeze(0)  # (1, L_seg)
        loudness_scores_tensor = torch.from_numpy(loudness_scores).to(self.device).float().unsqueeze(0)  # (1, L_seg)
        note_masks_tensor = torch.from_numpy(note_masks).to(self.device).float().unsqueeze(0)  # (1, L_seg)
        instrument_ids_tensor = torch.tensor([instrument_id], device=self.device)  # (1,)
        
        # f0_scoreを正規化
        f0_scores_normalized = normalize(f0_scores_tensor, self.f0_score_mean, self.f0_score_std).float()
        
        # note_maskを正規化
        note_masks_normalized = normalize(note_masks_tensor, self.note_mask_mean, self.note_mask_std).float()
        
        # Diffusion samplingでパラメータを生成
        self.logger.info(f"Sampling parameters from diffusion model")
        self.diffusion_model.eval()
        with torch.no_grad():
            params_sampled = self.diffuser.sample(
                model=self.diffusion_model,
                shape=(1, L_seg, 18),
                instrument_ids=instrument_ids_tensor,
                f0_score=f0_scores_normalized,
                guidance_scale=self.cfg_guidance_scale,
                note_mask=note_masks_normalized,
            )  # (1, L_seg, 18)
        
        # パラメータをsplit
        loudness_diff_mean_for_split = self.loudness_diff_mean if self.loudness_diff_mean is not None else self.loudness_mean
        loudness_diff_std_for_split = self.loudness_diff_std if self.loudness_diff_std is not None else self.loudness_std
        
        all_f0s, f0_diffs, all_loudnesses, all_loudness_diffs, all_z_features = split_params(
            aggregated=params_sampled,
            f0_mean=self.f0_mean,
            f0_std=self.f0_std,
            z_feature_mean=self.z_feature_mean,
            z_feature_std=self.z_feature_std,
            loudness_diff_mean=loudness_diff_mean_for_split,
            loudness_diff_std=loudness_diff_std_for_split,
            loudness_mean=self.loudness_mean,
            loudness_std=self.loudness_std,
            loudness_score=loudness_scores_tensor,
            use_loudness_diff=self.use_loudness_diff,
            f0_diff_mean=self.f0_diff_mean if self.f0_diff_mean is not None else None,
            f0_diff_std=self.f0_diff_std if self.f0_diff_std is not None else None,
            f0_score=f0_scores_tensor,
            use_f0_diff=self.use_f0_diff,
            allowed_f0_range=None,
            allowed_loudness_range=None,
        )
        
        # numpyに変換
        all_f0s_np = all_f0s.squeeze(0).detach().cpu().numpy()  # (L_seg,)
        all_loudnesses_np = all_loudnesses.squeeze(0).detach().cpu().numpy()  # (L_seg,)
        all_z_features_np = all_z_features.squeeze(0).detach().cpu().numpy()  # (L_seg, 16)
        
        # cent to Hz変換（f0）
        from api.models.diffusion.pitch import cent_to_hz
        all_f0s_hz = cent_to_hz(all_f0s_np)
        
        return {
            "pitch": all_f0s_hz.tolist(),
            "loudness": all_loudnesses_np.tolist(),
            "z_feature": all_z_features_np.tolist(),
        }

