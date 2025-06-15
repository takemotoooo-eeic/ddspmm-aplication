import io
import json
from typing import Generator

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
from api.models.ddsp.loss import Loss, LossInputs
from api.models.midi_aligner.midi_aligner import AlignedMidi
from .load_model import load_model
from api.controllers.backend_api.openapi import models
from .model import DDSP, DDSP_Decoder, Z_Encoder

MODEL_WEIGHTS_PATH = "api/models/ddsp/weights/best_model.pth"
PRETRAIN_CONFIG_PATH = "api/config/pretrain.config.yaml"
PREPROCESS_CONFIG_PATH = "api/config/preprocess.config.yaml"
TRAIN_CONFIG_PATH = "api/config/train.config.yaml"
LOUDNESS_PATH = "api/models/ddsp/statistics/loudness.json"


class TrainInput(BaseModel):
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"device: {self.device}")
        self.model: DDSP = load_model(MODEL_WEIGHTS_PATH, self.device, model_config)
        self.logger.info(f"finished load_model")
        self.encoder = self.model.z_encoder
        self.decoder = self.model.decoder
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        self.logger.info(f"decoder_params: {decoder_params}")
        self.model.to(self.device)

    def _initialize_input(
        self,
        encoder: Z_Encoder,
        train_input: TrainInput,
        preprocess_config: PreprocessConfig,
        mean_loudness: float,
        std_loudness: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signal_mix, _, loudness_mix = preprocess_wav_file(
            train_input.wav_file, preprocess_config, self.device
        )
        z: torch.Tensor = encoder(signal_mix.unsqueeze(0))
        z = z.squeeze(0)
        z_features, pitches, loudnesses = [], [], []
        for midi in train_input.midi:
            pitch: torch.Tensor
            loudness: torch.Tensor
            mask: torch.Tensor
            pitch, loudness, mask = convert_midi_to_features(
                midi=midi,
                loudness=loudness_mix,
                sampling_rate=preprocess_config.sampling_rate,
                signal_length=signal_mix.shape[0],
                block_size=preprocess_config.block_size,
                device=self.device,
            )
            z_feature = torch.randn(pitch.shape[0], 16, device=self.device).float()
            z_mean = torch.mean(z[mask == 0], dim=0)
            z_feature[mask == 0] = z_mean
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
        train_config: TrainConfig,
        loss_config: LossConfig,
        preprocess_config: PreprocessConfig,
        num_instruments: int,
        mean_loudness: float,
        std_loudness: float,
        instrument_names: list[str],
    ) -> models.TrainDDSPOutputStream:
        optimizer = torch.optim.Adam(
            [z_features, pitches, loudnesses], lr=train_config.lr
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[2000, 3000], gamma=0.1
        )
        loss_fn = Loss(self.device, loss_config)
        epochs: int = train_config.epochs
        print(f"epochs: {epochs}")
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
            )
            loss: torch.Tensor = loss_fn(loss_inputs)
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": loss.item()})
            wandb.log({"loss": loss.item()}, step=epoch)
            if (epoch + 1) % 100 == 0:
                for i in range(num_instruments):
                    wandb.log(
                        {
                            f"instrument_{i}/generated_audio": wandb.Audio(
                                signals[i].reshape(-1).detach().cpu().numpy(),
                                sample_rate=preprocess_config.sampling_rate,
                            ),
                        },
                        step=epoch,
                    )
            
            # 10エポックごとに進捗状況を返す
            if (epoch + 1) % 10 == 0:
                yield models.TrainingProgress(
                    current_epoch=epoch + 1,
                    total_epochs=epochs,
                    loss=float(loss.item())
                )
        
        wandb.finish()

        loudnesses = loudnesses * std_loudness + mean_loudness

        yield models.Features(
            features=[
                models.Feature(
                    instrument_name=instrument_names[i],
                    z_feature=z_features[i].reshape(-1,16).detach().cpu().numpy().tolist(),
                    pitch=pitches[i].reshape(-1).detach().cpu().numpy().tolist(),
                    loudness=loudnesses[i].reshape(-1).detach().cpu().numpy().tolist(),
                )
                for i in range(num_instruments)
            ],
        )

    def train(self, train_input: TrainInput) -> models.TrainDDSPOutputStream:
        self.logger.info("Start training...")

        train_config = TrainConfig.from_config_path(TRAIN_CONFIG_PATH)
        loss_config = LossConfig.from_config_path(TRAIN_CONFIG_PATH)
        preprocess_config = PreprocessConfig.from_config_path(PREPROCESS_CONFIG_PATH)

        wandb.init(
            project="si-ddspmm-sync-api",
            name="train",
            config={
                "train": train_config.model_dump(),
                "loss": loss_config.model_dump(),
            },
        )
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
            train_config=train_config,
            loss_config=loss_config,
            preprocess_config=preprocess_config,
            num_instruments=train_input.num_instruments,
            mean_loudness=mean_loudness,
            std_loudness=std_loudness,
            instrument_names=train_input.instrument_names,
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
