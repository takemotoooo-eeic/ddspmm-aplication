import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from api.libs.audio_splice import (
    match_segment_loudness,
    read_wav_bytes,
    splice_region,
    validate_segment,
    write_wav_bytes,
)
from api.libs.logging import get_logger

MELODYFLOW_TARGET_SR = 48_000
MELODYFLOW_TARGET_CHANNELS = 2
DEFAULT_MODEL_NAME = "facebook/melodyflow-t24-30secs"
DEFAULT_LOCAL_MODEL_DIR = (
    Path(__file__).resolve().parent / "melodyflow_assets" / "melodyflow-t24-30secs"
)
REQUIRED_WEIGHT_FILES = ("state_dict.bin", "compression_state_dict.bin")


@dataclass
class MelodyFlowEditParams:
    start_sec: float
    end_sec: float
    text: str
    solver: str = "euler"
    steps: int = 125
    target_flowstep: float = 0.0
    regularize: bool = True
    regularization_strength: float = 0.2


class MelodyFlowModel:
    """MelodyFlow (audiocraft) による区間テキスト編集。"""

    def __init__(self, model_name: Optional[str] = None):
        self.logger = get_logger()
        self.model_name = model_name or _resolve_pretrained_path()
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from audiocraft.models import MelodyFlow
        except ImportError as e:
            raise RuntimeError(
                "audiocraft (MelodyFlow) がインストールされていません。"
                " `cd backend && uv sync` を実行してください。"
            ) from e

        if os.path.isdir(self.model_name):
            missing = [
                f
                for f in REQUIRED_WEIGHT_FILES
                if not os.path.isfile(os.path.join(self.model_name, f))
            ]
            if missing:
                raise RuntimeError(
                    f"MelodyFlow ローカル重みが不完全です: {self.model_name} "
                    f"(不足: {', '.join(missing)}). "
                    "`bash backend/scripts/download_melodyflow_model.sh` を実行してください。"
                )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(
            f"Loading MelodyFlow from: {self.model_name} (device={device}). "
            "初回は T5 (t5-base) の Hugging Face 取得と約 4GB の重み読込があり、数分かかります。"
        )
        t0 = time.monotonic()
        try:
            # audiocraft の MelodyFlow は nn.Module ではなく .to() を持たない。
            # デバイスは get_pretrained(..., device=...) で指定する。
            self._model = MelodyFlow.get_pretrained(self.model_name, device=device)
        except Exception:
            self.logger.exception(
                "MelodyFlow のロードに失敗しました。"
                " T5 の未取得・中断の場合はホストで "
                "`bash backend/scripts/download_melodyflow_t5.sh` を実行し、"
                "`backend/.cache/huggingface` をマウントしたうえで api を再起動してください。"
            )
            raise
        self.logger.info(
            f"MelodyFlow: load finished in {time.monotonic() - t0:.1f}s"
        )
        return self._model

    def _segment_to_tensor(
        self, segment: np.ndarray, sr: int, model
    ) -> torch.Tensor:
        from audiocraft.data.audio_utils import convert_audio

        if segment.ndim == 1:
            wav = torch.from_numpy(segment).float().unsqueeze(0)
        else:
            wav = torch.from_numpy(segment.T).float()

        if wav.dim() == 2:
            wav = wav.unsqueeze(0)

        wav = convert_audio(
            wav,
            sr,
            MELODYFLOW_TARGET_SR,
            MELODYFLOW_TARGET_CHANNELS,
        )
        max_samples = int(MELODYFLOW_TARGET_SR * model.duration)
        if wav.shape[-1] > max_samples:
            wav = wav[..., :max_samples]
        return wav

    def _tensor_to_numpy(self, wav: torch.Tensor, num_samples: int) -> np.ndarray:
        out = wav.detach().cpu().float()
        if out.dim() == 3:
            out = out[0]
        if out.dim() == 2:
            out = out.T.numpy()
        else:
            out = out.numpy()
        if out.shape[0] > num_samples:
            out = out[:num_samples]
        elif out.shape[0] < num_samples:
            out = np.pad(
                out,
                ((0, num_samples - out.shape[0]), (0, 0)),
                mode="constant",
            )
        return out.astype(np.float32)

    def _edit_segment(
        self,
        segment: np.ndarray,
        sr: int,
        text: str,
        params: MelodyFlowEditParams,
    ) -> np.ndarray:
        model = self._load_model()
        duration = min(params.end_sec - params.start_sec, model.duration)

        model.set_generation_params(
            solver=params.solver,
            steps=params.steps,
            duration=duration,
        )
        model.set_editing_params(
            solver=params.solver,
            steps=params.steps,
            target_flowstep=params.target_flowstep,
            regularize=params.regularize,
            lambda_kl=params.regularization_strength,
        )

        wav = self._segment_to_tensor(segment, sr, model)
        prompt_tokens = model.encode_audio(wav.to(model.device))
        self.logger.info(f"device: {model.device}")

        outputs = model.edit(
            prompt_tokens=prompt_tokens,
            descriptions=[text],
            src_descriptions=[""],
            progress=False,
            return_tokens=False,
        )
        seg_samples = segment.shape[0] if segment.ndim == 2 else len(segment)
        return self._tensor_to_numpy(outputs[0], seg_samples)

    def edit_region(self, wav_bytes: bytes, params: MelodyFlowEditParams) -> bytes:
        full, sr = read_wav_bytes(wav_bytes)
        duration_sec = full.shape[0] / sr
        validate_segment(duration_sec, params.start_sec, params.end_sec)

        start = int(round(params.start_sec * sr))
        end = int(round(params.end_sec * sr))
        segment = full[start:end]
        if segment.ndim == 2 and segment.shape[1] == 1:
            segment = segment[:, 0]

        edited_segment = self._edit_segment(segment, sr, params.text, params)
        edited_segment = match_segment_loudness(segment, edited_segment)
        result = splice_region(full, sr, params.start_sec, params.end_sec, edited_segment)
        return write_wav_bytes(result, sr)


def _local_model_dir() -> Path:
    override = os.environ.get("MELODYFLOW_MODEL_DIR")
    if override:
        return Path(override)
    return DEFAULT_LOCAL_MODEL_DIR


def _has_local_weights(model_dir: Path) -> bool:
    return all((model_dir / name).is_file() for name in REQUIRED_WEIGHT_FILES)


def _resolve_pretrained_path() -> str:
    """HF Hub ID またはローカルディレクトリ。ローカルがあれば優先。"""
    explicit = os.environ.get("MELODYFLOW_MODEL")
    if explicit:
        return explicit
    local = _local_model_dir()
    if _has_local_weights(local):
        return str(local)
    return DEFAULT_MODEL_NAME


_melodyflow_model: Optional[MelodyFlowModel] = None


def get_melodyflow_model() -> MelodyFlowModel:
    global _melodyflow_model
    if _melodyflow_model is None:
        _melodyflow_model = MelodyFlowModel()
    return _melodyflow_model
