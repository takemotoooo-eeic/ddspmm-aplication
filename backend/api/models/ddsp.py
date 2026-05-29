import io
import json
import os

import numpy as np
import soundfile
import torch
from tqdm import tqdm
import wandb
from pydantic import BaseModel

from api.config import LossConfig, ModelConfig, PreprocessConfig, TrainConfig
from api.libs.logging import get_logger
from api.libs.midi import convert_midi_to_features
from api.libs.wav import preprocess_wav_file, reshape_to_segments
from api.models.loss import Loss, LossInputs
from api.libs.note import AlignedMidi
from api.controllers.backend_api.openapi import models
from api.libs.const import MODEL_WEIGHTS_PATH, PRETRAIN_CONFIG_PATH, PREPROCESS_CONFIG_PATH, TRAIN_CONFIG_PATH, LOUDNESS_PATH


from .model import DDSP, DDSP_Decoder, Z_Encoder

_ddsp_model: "DDSPModel | None" = None
_ddsp_decoder: DDSP_Decoder | None = None
_ddsp_device: torch.device | None = None


def _resolve_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_decoder_weights(decoder: DDSP_Decoder, device: torch.device) -> None:
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    decoder_state = {
        key[len("decoder.") :]: value
        for key, value in state_dict.items()
        if key.startswith("decoder.")
    }
    decoder.load_state_dict(decoder_state, strict=False)


def get_ddsp_decoder() -> DDSP_Decoder:
    """Diffusion ガイダンス用。エンコーダなしで decoder のみ GPU に1回だけ載せる。"""
    global _ddsp_decoder, _ddsp_device, _ddsp_model
    device = _resolve_device()
    if _ddsp_model is not None:
        return _ddsp_model.decoder
    if _ddsp_decoder is not None and _ddsp_device == device:
        return _ddsp_decoder

    logger = get_logger()
    model_config: ModelConfig = ModelConfig.from_config_path(PRETRAIN_CONFIG_PATH)
    decoder = DDSP_Decoder(
        hidden_size=model_config.hidden_size,
        n_harmonic=model_config.n_harmonic,
        n_bands=model_config.n_bands,
        sampling_rate=model_config.sampling_rate,
        block_size=model_config.block_size,
    ).to(device)
    _load_decoder_weights(decoder, device)
    logger.info("DDSP decoder loaded for diffusion guidance")
    _ddsp_decoder = decoder
    _ddsp_device = device
    return decoder


def get_ddsp_model() -> "DDSPModel":
    """DDSP train / generate 用。プロセス内で DDSPModel を1つだけ保持する。"""
    global _ddsp_model, _ddsp_decoder, _ddsp_device
    if _ddsp_model is None:
        _ddsp_model = DDSPModel()
        _ddsp_decoder = _ddsp_model.decoder
        _ddsp_device = _ddsp_model.device
    return _ddsp_model


class TrainInput(BaseModel):
    epochs: int = 1000
    lr: float = 0.1
    wav_file: bytes
    num_instruments: int
    instrument_names: list[str]
    midi: list[AlignedMidi]


class Feature(BaseModel):
    instrument_name: str
    pitch: list[float]
    loudness: list[float]
    z_feature: list[list[float]]


class TrainOutput(BaseModel):
    features: list[Feature]


class DDSPModel:
    def __init__(
        self,
    ):
        self.logger = get_logger()
        model_config: ModelConfig = ModelConfig.from_config_path(PRETRAIN_CONFIG_PATH)
        self.device = _resolve_device()
        self.logger.info(f"device: {self.device}")
        self.model: DDSP = self._load_model(MODEL_WEIGHTS_PATH, self.device, model_config)
        self.logger.info(f"finished load_model")
        self.encoder = self.model.z_encoder
        self.decoder = self.model.decoder
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        self.logger.info(f"decoder_params: {decoder_params}")
        self.model.to(self.device)

    def _load_model(
        self, model_path: str, device: torch.device, model_config: ModelConfig
    ) -> DDSP:
        model = DDSP(
            hidden_size=model_config.hidden_size,
            n_harmonic=model_config.n_harmonic,
            n_bands=model_config.n_bands,
            sampling_rate=model_config.sampling_rate,
            block_size=model_config.block_size,
        ).to(device)

        old_state_dict = torch.load(model_path, map_location=device)

        model.load_state_dict(old_state_dict, strict=False)

        self.logger.info(f"model loaded: {model_path}")
        self.logger.info(f"parameters: {sum(p.numel() for p in model.parameters())}")
        return model

    def _initialize_input(
        self,
        encoder: Z_Encoder,
        train_input: TrainInput,
        preprocess_config: PreprocessConfig,
        mean_loudness: float,
        std_loudness: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signal_mix, _, _ = preprocess_wav_file(
            train_input.wav_file, preprocess_config, self.device
        )
        z: torch.Tensor = encoder(signal_mix.unsqueeze(0))
        z = z.squeeze(0)
        z_features, pitches, loudnesses = [], [], []
        for midi in train_input.midi:
            pitch: torch.Tensor
            loudness: torch.Tensor
            pitch, loudness = convert_midi_to_features(
                midi=midi,
                sampling_rate=preprocess_config.sampling_rate,
                signal_length=signal_mix.shape[0],
                device=self.device,
                block_size=preprocess_config.block_size,
            )
            z_feature = torch.randn(pitch.shape[0], 16, device=self.device).float()
            output = reshape_to_segments(
                {
                    "signal": signal_mix,
                    "pitch": pitch,
                    "loudness": loudness,
                    "z_feature": z_feature,
                },
                preprocess_config.signal_length,
            )
            loudness = output["loudness"].unsqueeze(-1)
            pitch = output["pitch"].unsqueeze(-1)
            signal_mix_reshaped = output["signal"]
            z_feature = output["z_feature"]
            z_features.append(z_feature)
            pitches.append(pitch)
            loudnesses.append(loudness)

        pitches = torch.stack(pitches).to(self.device).detach().requires_grad_(True)
        loudnesses = torch.stack(loudnesses).to(self.device)
        loudnesses = (loudnesses - mean_loudness) / std_loudness
        loudnesses = loudnesses.detach().requires_grad_(True)
        z_features = (
            torch.stack(z_features).to(self.device).detach().requires_grad_(True)
        )
        return signal_mix_reshaped, z_features, pitches, loudnesses

    def _train(
        self,
        model: DDSP_Decoder,
        reference_audio: torch.Tensor,
        z_features: torch.Tensor,
        pitches: torch.Tensor,
        loudnesses: torch.Tensor,
        loss_config: LossConfig,
        preprocess_config: PreprocessConfig,
        num_instruments: int,
        mean_loudness: float,
        std_loudness: float,
        instrument_names: list[str],
        epochs: int,
        lr: float,
        aligned_midi_list: list[AlignedMidi],
    ) -> models.Features:
        optimizer = torch.optim.Adam(
            [z_features, pitches, loudnesses], lr=lr
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[2000, 3000], gamma=0.1
        )
        instrument_names_fixed = ["ob", "vc"]
        loss_fn = Loss(self.device, loss_config, instrument_names_fixed)
        self.logger.info(f"epochs: {epochs}")

        pbar = tqdm(range(epochs), desc="Training")
        for epoch in pbar:
            optimizer.zero_grad()
            signals = []
            for i in range(num_instruments):
                signal, *_ = model(pitches[i], loudnesses[i], z_features[i])
                signal = signal.squeeze(-1)
                signals.append(signal)

            signal_mix = torch.stack(signals)
            signal_mix = signal_mix.sum(dim=0).squeeze(0)

            loss_inputs = LossInputs.from_results(
                loss_config=loss_config,
                signal_pred=signal_mix,
                signal_target=reference_audio,
                loudness=loudnesses,
                pitch=pitches,
                z_feature=z_features,
                instrument_names=instrument_names,
            )
            loss: torch.Tensor = loss_fn(loss_inputs)
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": loss.item()})

        loudnesses = loudnesses * std_loudness + mean_loudness

        return models.Features(
            features=[
                models.Feature(
                    instrument_name=instrument_names[i],
                    z_feature=z_features[i].reshape(-1,16).detach().cpu().numpy().tolist(),
                    pitch=pitches[i].reshape(-1).detach().cpu().numpy().tolist(),
                    loudness=loudnesses[i].reshape(-1).detach().cpu().numpy().tolist(),
                    notes=[
                        models.Note(
                            start=note.start,
                            frequency=note.frequency,
                            duration=note.duration,
                        )
                        for note in aligned_midi_list[i].notes
                    ],
                )
                for i in range(num_instruments)
            ],
        )

    def train(self, train_input: TrainInput) -> models.Features:
        self.logger.info("Start training...")

        train_config = TrainConfig.from_config_path(TRAIN_CONFIG_PATH)
        loss_config = LossConfig.from_config_path(TRAIN_CONFIG_PATH)
        preprocess_config = PreprocessConfig.from_config_path(PREPROCESS_CONFIG_PATH)

        with open(LOUDNESS_PATH, "r") as f:
            loudness_config = json.load(f)
        mean_loudness = loudness_config["mean"]
        std_loudness = loudness_config["std"]

        reference_audio, z_features, pitches, loudnesses = self._initialize_input(
            encoder=self.encoder,
            train_input=train_input,
            preprocess_config=preprocess_config,
            mean_loudness=mean_loudness,
            std_loudness=std_loudness,
        )
        self.logger.info(f"finished initialize_input")
        return self._train(
            model=self.decoder,
            reference_audio=reference_audio,
            z_features=z_features,
            pitches=pitches,
            loudnesses=loudnesses,
            loss_config=loss_config,
            preprocess_config=preprocess_config,
            num_instruments=train_input.num_instruments,
            mean_loudness=mean_loudness,
            std_loudness=std_loudness,
            instrument_names=train_input.instrument_names,
            epochs=train_config.epochs,
            lr=train_config.lr,
            aligned_midi_list=train_input.midi,
        )

    def generate(
        self,
        pitch: list[float],
        loudness: list[float],
        z_feature: list[list[float]],
    ) -> bytes:
        preprocess_config = PreprocessConfig.from_config_path(PREPROCESS_CONFIG_PATH)
        with open(LOUDNESS_PATH, "r") as f:
            loudness_config = json.load(f)
        mean_loudness = loudness_config["mean"]
        std_loudness = loudness_config["std"]

        pitch_i = torch.tensor(pitch, dtype=torch.float32, device=self.device)
        loudness_i = torch.tensor(loudness, dtype=torch.float32, device=self.device)
        z_feature_i = torch.tensor(z_feature, dtype=torch.float32, device=self.device)
        mock_signal = torch.randn(pitch_i.shape[0] * preprocess_config.block_size, device=self.device)
        output = reshape_to_segments(
            {
                "signal": mock_signal,
                "pitch": pitch_i,
                "loudness": loudness_i,
                "z_feature": z_feature_i,
            },
            preprocess_config.signal_length,
        )
        pitch_i = output["pitch"].unsqueeze(-1)
        loudness_i = output["loudness"].unsqueeze(-1)
        z_feature_i = output["z_feature"]

        loudness_i = (loudness_i - mean_loudness) / std_loudness

        with torch.no_grad():
            signal: torch.Tensor
            signal, *_ = self.decoder(pitch_i, loudness_i, z_feature_i)
            signal = signal.squeeze(-1)
        buffer = io.BytesIO()
        signal_np = signal.reshape(-1).detach().cpu().numpy().astype(np.float32)
        soundfile.write(
            buffer, signal_np, samplerate=preprocess_config.sampling_rate, format="WAV"
        )
        buffer.seek(0)
        return buffer.read()
