from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm

from api.libs.core import normalize, denormalize
from api.libs.instrument import Instrument

from api.libs.core import (
    amp_to_impulse_response,
    fft_convolve,
    gru,
    harmonic_synth,
    mlp,
    remove_above_nyquist,
    scale_function,
    upsample,
)


class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x


class Z_Encoder(nn.Module):
    def __init__(
        self,
        n_fft,
        hop_length,
        sample_rate=16000,
        n_mels=128,
        n_mfcc=30,
        gru_units=512,
        z_units=16,
        bidirectional=False,
    ):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=dict(
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=20.0,
                f_max=8000.0,
            ),
        )

        self.norm = nn.InstanceNorm1d(n_mfcc, affine=True)
        self.permute = lambda x: x.permute(0, 2, 1)
        self.gru = nn.GRU(
            input_size=n_mfcc,
            hidden_size=gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dense = nn.Linear(gru_units * 2 if bidirectional else gru_units, z_units)

    def forward(self, batch):
        x = batch

        x = self.mfcc(x)
        x = x[:, :, :-1]
        x = self.norm(x)
        x = self.permute(x)
        x, _ = self.gru(x)
        x = self.dense(x)

        return x


class DDSP_Decoder(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate, block_size):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3)] * 2)
        self.in_mlps.extend([mlp(16, hidden_size, 3)])

        self.gru = gru(3, hidden_size)
        self.out_mlp = mlp(hidden_size + 2, hidden_size, 3)

        self.proj_matrices = nn.ModuleList(
            [
                nn.Linear(hidden_size, n_harmonic + 1),
                nn.Linear(hidden_size, n_bands),
            ]
        )

        self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def forward(self, pitch, loudness, z):
        hidden = torch.cat(
            [self.in_mlps[0](pitch), self.in_mlps[1](loudness), self.in_mlps[2](z)], -1
        )

        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = self.upsample_with_window(amplitudes, 192000)
        pitch = upsample(pitch, 512)

        harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = (
            torch.rand(
                impulse.shape[0],
                impulse.shape[1],
                self.block_size,
            ).to(impulse)
            * 2
            - 1
        )

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        # reverb part
        signal = self.reverb(signal)

        return signal, harmonic, noise, total_amp

    def upsample_with_window(self, input_amp, n_timesteps):
        for i in range(input_amp.shape[0]):
            amplitudes = input_amp[i]
            amplitudes = amplitudes.unsqueeze(0)
            amplitudes = torch.cat([amplitudes, amplitudes[:, -1:, :]], axis=1)

            n_frames = int(amplitudes.shape[1])
            n_intervals = n_frames - 1

            hop_size = n_timesteps // n_intervals

            window_size = hop_size * 2
            hann_window = torch.hann_window(window_size).to(amplitudes)

            x = amplitudes.unsqueeze(-1)

            x = x.permute(0, 2, 1, 3)

            hann_window = hann_window.unsqueeze(0)
            hann_window = hann_window.unsqueeze(0)
            hann_window = hann_window.unsqueeze(0)
            x_windowed = x * hann_window
            x_windowed = x_windowed.permute(0, 1, 3, 2)

            x = torch.nn.functional.fold(
                x_windowed[0],
                output_size=(1, n_timesteps + (hop_size * 2)),
                kernel_size=(1, window_size),
                stride=(1, hop_size),
            )

            x = x.permute(1, 2, 0, 3)
            x = x.squeeze(0)
            x = x.permute(0, 2, 1)
            x = x[:, hop_size:-hop_size, :]
            if i == 0:
                upsample = x
            else:
                upsample = torch.cat((upsample, x), 0)

        return upsample

    def realtime_forward(self, pitch, loudness):
        hidden = torch.cat(
            [
                self.in_mlps[0](pitch),
                self.in_mlps[1](loudness),
            ],
            -1,
        )

        gru_out, cache = self.gru(hidden, self.cache_gru)
        self.cache_gru.copy_(cache)

        hidden = torch.cat([gru_out, pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        n_harmonic = amplitudes.shape[-1]
        omega = torch.cumsum(2 * math.pi * pitch / self.sampling_rate, 1)

        omega = omega + self.phase
        self.phase.copy_(omega[0, -1, 0] % (2 * math.pi))

        omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)

        harmonic = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = (
            torch.rand(
                impulse.shape[0],
                impulse.shape[1],
                self.block_size,
            ).to(impulse)
            * 2
            - 1
        )

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        return signal


class DDSP(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate, block_size):
        super().__init__()
        self.n_fft = 2048
        self.z_encoder = Z_Encoder(n_fft=self.n_fft, hop_length=512)
        self.decoder = DDSP_Decoder(
            hidden_size, n_harmonic, n_bands, sampling_rate, block_size
        )

        # 互換性のための属性
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def forward(self, pitch, loudness, audio):
        z = self.z_encoder(audio)
        signal, harmonic, noise, total_amp = self.decoder(pitch, loudness, z)
        return signal, harmonic, noise, total_amp, z

def _pos_encoding(time_idx, output_dim, device="cpu"):
    """時間ステップの位置エンコーディング"""
    t, D = time_idx, output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D, device=device)
    div_term = torch.exp(i / D * math.log(10000))

    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])
    return v


def pos_encoding(timesteps, output_dim, device="cpu"):
    """バッチ単位の位置エンコーディング"""
    batch_size = len(timesteps)
    device = timesteps.device
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(timesteps[i], output_dim, device)
    return v


class FiLM(nn.Module):
    """Feature-wise Linear Modulation"""

    def __init__(self, cond_dim: int, out_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, out_dim)
        self.beta = nn.Linear(cond_dim, out_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) or (B, C, L) or (B, L, C)
               - 4次元: (B, C, H, W) - UNetのConvBlock
               - 3次元: (B, C, L) or (B, L, C) - チャネル次元の位置に応じて自動判定
            condition: (B, cond_dim)
        """
        gamma = self.gamma(condition)  # (B, out_dim)
        beta = self.beta(condition)  # (B, out_dim)

        # 次元に合わせてreshape
        if x.dim() == 4:
            # (B, C, H, W) の場合 - UNet
            gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
            beta = beta.view(beta.size(0), beta.size(1), 1, 1)
        elif x.dim() == 3:
            # 3次元テンソルの場合、チャネル次元の位置を自動判定
            B, dim1, dim2 = x.shape
            if dim1 == gamma.size(1):
                # (B, C, L) 形式 - チャネル次元が1番目
                gamma = gamma.view(B, gamma.size(1), 1)
                beta = beta.view(B, beta.size(1), 1)
            elif dim2 == gamma.size(1):
                # (B, L, C) 形式 - チャネル次元が2番目（DiT）
                gamma = gamma.unsqueeze(1)  # (B, 1, out_dim)
                beta = beta.unsqueeze(1)  # (B, 1, out_dim)
            else:
                raise ValueError(
                    f"FiLM: Cannot determine channel dimension. "
                    f"x.shape={x.shape}, gamma.size(1)={gamma.size(1)}"
                )

        return gamma * x + beta


class ConvBlock(nn.Module):
    """FiLM付きのConvolution Block (2D画像版)"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_embed_dim: int,
        film_cond_dim: Optional[int] = None,
        f0_score_scale: float = 1.0,
    ):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch), nn.ReLU(), nn.Linear(in_ch, in_ch)
        )

        self.f0_score_mlp = nn.Sequential(
            nn.Conv2d(1, in_ch, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
        )

        if film_cond_dim is not None:
            self.film = FiLM(film_cond_dim, out_ch)
        else:
            self.film = None
        
        self.f0_score_scale = f0_score_scale

    def forward(
        self,
        x: torch.Tensor,
        time_embed: torch.Tensor,
        film_cond: Optional[torch.Tensor] = None,
        f0_score: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: (B, C, H, W) - 2次元画像 (H=18特徴次元, W=L時間軸)
            time_embed: (B, time_embed_dim)
            film_cond: (B, film_cond_dim) or None
            f0_score: (B, W) or None - f0スコア（時間変動を保持、Wは時間軸の長さ）
        """
        B, C, H, W = x.shape

        # Time embedding
        v = self.time_mlp(time_embed)  # (B, in_ch)
        v = v.view(B, C, 1, 1)  # (B, in_ch, 1, 1)
        v = v.expand(B, C, H, W)  # (B, in_ch, H, W) - ブロードキャスト

        if f0_score is not None:
            f0_score_input = f0_score.unsqueeze(1).unsqueeze(1)
            f0_score_input = f0_score_input.expand(B, 1, H, W)
            f0_score_encoded = self.f0_score_mlp(f0_score_input)
            v = v + self.f0_score_scale * f0_score_encoded

        x = x + v

        # Convolution
        y = self.convs(x)

        # FiLM conditioning
        if self.film is not None and film_cond is not None:
            y = self.film(y, film_cond)

        return y
    


class UNetDiffusion(nn.Module):
    """合成パラメータ生成用のUNetベースのDiffusionモデル（2次元画像版）"""

    def __init__(
        self,
        time_embed_dim: int = 128,
        num_instruments: int = 10,
        film_cond_dim: int = 64,
        down1_out_ch: int = 64,
        down2_out_ch: int = 128,
        down3_out_ch: int = 256,
        bot1_out_ch: int = 512,
        cfg_dropout_prob: float = 0.1,
        f0_score_scale: float = 1.0,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.cfg_dropout_prob = cfg_dropout_prob

        # Instrument embedding for FiLM
        self.instrument_emb = nn.Embedding(num_instruments, film_cond_dim)
        
        # Null embedding for CFG (Classifier-Free Guidance)
        # 学習可能なnull embedding（条件なしの場合に使用）
        self.null_instrument_emb = nn.Parameter(torch.zeros(1, film_cond_dim))

        # 入力チャネル数は1（2次元画像として扱う）
        # Downsampling path
        self.down1 = ConvBlock(1, down1_out_ch, time_embed_dim, film_cond_dim, f0_score_scale)
        self.down2 = ConvBlock(
            down1_out_ch, down2_out_ch, time_embed_dim, film_cond_dim, f0_score_scale
        )
        self.down3 = ConvBlock(
            down2_out_ch, down3_out_ch, time_embed_dim, film_cond_dim, f0_score_scale
        )
        self.bot1 = ConvBlock(down3_out_ch, bot1_out_ch, time_embed_dim, film_cond_dim, f0_score_scale)

        # Upsampling path
        self.up3 = ConvBlock(
            bot1_out_ch + down3_out_ch, down3_out_ch, time_embed_dim, film_cond_dim, f0_score_scale
        )
        self.up2 = ConvBlock(
            down3_out_ch + down2_out_ch, down2_out_ch, time_embed_dim, film_cond_dim, f0_score_scale
        )
        self.up1 = ConvBlock(
            down2_out_ch + down1_out_ch, down1_out_ch, time_embed_dim, film_cond_dim, f0_score_scale
        )
        self.out = nn.Conv2d(down1_out_ch, 1, kernel_size=1)

        # 高さ（特徴次元）は固定、時間軸のみダウンサンプリング
        self.pool = nn.MaxPool2d(
            kernel_size=(1, 2), stride=(1, 2)
        )  # 高さ方向は1、幅方向のみ2
        self.upsample = nn.Upsample(
            scale_factor=(1, 2), mode="bilinear", align_corners=False
        )  # 高さ方向は1、幅方向のみ2倍

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        instrument_ids: torch.Tensor,
        f0_score: torch.Tensor,
        instrument_cond_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            x: (B, L, H) - ノイズが加えられた合成パラメータ（Hはheight_dim）
            timesteps: (B,) - 時間ステップ
            instrument_ids: (B,) - 楽器ID
            f0_score: (B, L) - f0スコア（conditioning用）
            instrument_cond_mask: (B,) or None - CFG用の条件マスク（True=条件付き、False=条件なし）
                                     Noneの場合は学習時にランダムにドロップアウト
        """
        B = x.shape[0]
        
        # Time embedding
        t = pos_encoding(timesteps, self.time_embed_dim, device=x.device)

        # Instrument embedding with CFG dropout
        if instrument_cond_mask is None:
            # 学習時: ランダムにCFG dropoutを適用
            if self.training and self.cfg_dropout_prob > 0:
                instrument_cond_mask = torch.rand(B, device=x.device) > self.cfg_dropout_prob
            else:
                # 推論時（maskが指定されていない場合）: 常に条件付き
                instrument_cond_mask = torch.ones(B, device=x.device, dtype=torch.bool)
        else:
            instrument_cond_mask = instrument_cond_mask.to(x.device)
        
        # 条件付き/条件なしでembeddingを切り替え
        instrument_emb_cond = self.instrument_emb(instrument_ids)  # (B, film_cond_dim)
        instrument_emb_uncond = self.null_instrument_emb.expand(B, -1)  # (B, film_cond_dim)
        
        # maskに応じて選択: Trueの場合は条件付き、Falseの場合は条件なし
        film_cond = torch.where(
            instrument_cond_mask.unsqueeze(-1),
            instrument_emb_cond,
            instrument_emb_uncond,
        )  # (B, film_cond_dim)

        # (B, L, H) -> (B, 1, H, L) - 2次元画像として扱う
        # 高さ=H（特徴次元）、幅=L（時間軸）、チャネル=1
        _, L, _ = x.shape
        original_L = L
        
        # MaxPool2dとUpsampleの整合性のため、Lは8の倍数を想定
        if L % 8 != 0:
            pad_length = 8 - (L % 8)
            # 最後の値を複製してパディング
            x = torch.cat([x, x[:, -1:, :].expand(-1, pad_length, -1)], dim=1)
            L = x.shape[1]
            # f0_scoreも同様にパディング
            f0_score = torch.cat([f0_score, f0_score[:, -1:].expand(-1, pad_length)], dim=1)
        
        x = x.transpose(1, 2)  # (B, 18, L)
        x = x.unsqueeze(1)  # (B, 1, 18, L)

        # Downsampling（高さは18で固定、時間軸のみダウンサンプリング）
        x1 = self.down1(x, t, film_cond, f0_score)  # (B, down1_out_ch, 18, L)
        x = self.pool(x1)  # (B, down1_out_ch, 18, L/2)
        # f0_scoreもダウンサンプリング（時間軸方向のみ）
        # f0_score: (B, L) -> (B, 1, 1, L) -> interpolate -> (B, 1, 1, L/2) -> (B, L/2)
        f0_score_down1 = (
            F.interpolate(
                f0_score.unsqueeze(1).unsqueeze(1),
                size=(1, x.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
            .squeeze(1)
        )  # (B, L/2)

        x2 = self.down2(x, t, film_cond, f0_score_down1)  # (B, down2_out_ch, 18, L/2)
        x = self.pool(x2)  # (B, down2_out_ch, 18, L/4)
        f0_score_down2 = (
            F.interpolate(
                f0_score_down1.unsqueeze(1).unsqueeze(1),
                size=(1, x.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
            .squeeze(1)
        )  # (B, L/4)

        x3 = self.down3(x, t, film_cond, f0_score_down2)  # (B, down3_out_ch, 18, L/4)
        x = self.pool(x3)  # (B, down3_out_ch, 18, L/8)
        f0_score_down3 = (
            F.interpolate(
                f0_score_down2.unsqueeze(1).unsqueeze(1),
                size=(1, x.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
            .squeeze(1)
        )  # (B, L/8)

        # Bottleneck
        x = self.bot1(x, t, film_cond, f0_score_down3)  # (B, bot1_out_ch, 18, L/8)

        # Upsampling（高さは18で固定、時間軸のみアップサンプリング）
        x = self.upsample(x)  # (B, bot1_out_ch, 18, L/4)
        x = torch.cat([x, x3], dim=1)  # (B, bot1_out_ch + down3_out_ch, 18, L/4)
        x = self.up3(x, t, film_cond, f0_score_down2)  # (B, down3_out_ch, 18, L/4)

        x = self.upsample(x)  # (B, down3_out_ch, 18, L/2)
        x = torch.cat([x, x2], dim=1)  # (B, down3_out_ch + down2_out_ch, 18, L/2)
        x = self.up2(x, t, film_cond, f0_score_down1)  # (B, down2_out_ch, 18, L/2)

        x = self.upsample(x)  # (B, down2_out_ch, 18, L)
        x = torch.cat([x, x1], dim=1)  # (B, down2_out_ch + down1_out_ch, 18, L)
        x = self.up1(x, t, film_cond, f0_score)  # (B, down1_out_ch, 18, L)

        # Output
        x = self.out(x)  # (B, 1, 18, L)

        # (B, 1, 18, L) -> (B, L, 18)
        x = x.squeeze(1)  # (B, 18, L)
        x = x.transpose(1, 2)  # (B, L, 18)

        if x.shape[1] > original_L:
            x = x[:, :original_L, :]
        
        return x


class Diffuser:
    """Diffusionプロセスを管理するクラス"""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.scheduler = None

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor):
        """
        ノイズを追加

        Args:
            x_0: (B, L, D) - 元の合成パラメータ
            t: (B,) - 時間ステップ (1以上、1-indexed)
        Returns:
            x_t: (B, L, D) - ノイズが加えられた合成パラメータ
            noise: (B, L, D) - 追加されたノイズ
        """
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alpha_bars[0] is for t=1
        alpha_bar = self.alpha_bars[t_idx]  # (B,)
        alpha_bar = alpha_bar.view(alpha_bar.size(0), 1, 1)  # (B, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        instrument_ids: torch.Tensor,
        f0_score: torch.Tensor,
        prev_t: int | None = None,
    ):
        """
        ノイズ除去（サンプリング時）

        Args:
            model: UNetDiffusionモデル
            x: (B, L, D) - ノイズが加えられた合成パラメータ
            t: (B,) - 時間ステップ (1-indexed)
            instrument_ids: (B,) - 楽器ID
            f0_score: (B, L) - f0スコア
            prev_t: 遷移先の時間ステップ。Noneの場合は t-1
        Returns:
            x_prev: (B, L, D) - 前のステップのサンプル
        """
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1
        alpha_bar = self.alpha_bars[t_idx]  # (N,)

        if prev_t is None:
            prev_t_tensor = torch.clamp(t - 1, min=0)
        else:
            prev_t_tensor = torch.full_like(t, prev_t)

        alpha_bar_prev = torch.where(
            prev_t_tensor > 0,
            self.alpha_bars[prev_t_tensor - 1],
            torch.ones_like(alpha_bar),
        )
        alpha = alpha_bar / alpha_bar_prev
        beta = 1 - alpha
        mask = (prev_t_tensor > 0).float()

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1)
        beta = beta.view(N, 1, 1)

        with torch.no_grad():
            eps = model(x, t, instrument_ids, f0_score)

        # 予測されたノイズから x_0 を推定
        pred_x_0 = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)

        # prev_tへのDDPM遷移。prev_t=t-1なら通常の1ステップDDPMと一致する。
        mu = (torch.sqrt(alpha_bar_prev) * beta / (1 - alpha_bar)) * pred_x_0 + (
            torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)
        ) * x

        std = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        std = std.view(N, 1, 1)

        noise = torch.randn_like(x, device=self.device)
        noise = noise * mask.view(N, 1, 1)  # t=1の場合はノイズを追加しない

        return mu + noise * std

    def denoise_ddim(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        instrument_ids: torch.Tensor,
        f0_score: torch.Tensor,
        prev_t: int,
    ):
        """DDIM (eta=0) による決定論的デノイジング。少ステップサンプリング向け。"""
        t_idx = t - 1
        alpha_bar = self.alpha_bars[t_idx]  # (N,)
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1)

        if prev_t > 0:
            alpha_bar_prev = self.alpha_bars[prev_t - 1].expand(N).view(N, 1, 1)
        else:
            alpha_bar_prev = torch.ones(N, 1, 1, device=self.device)

        with torch.no_grad():
            eps = model(x, t, instrument_ids, f0_score)

        pred_x_0 = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
        return torch.sqrt(alpha_bar_prev) * pred_x_0 + torch.sqrt(1 - alpha_bar_prev) * eps

    @staticmethod
    def _build_sampling_timesteps(max_steps: int, steps: int) -> list[int]:
        if steps == max_steps:
            return list(range(max_steps, 0, -1))
        timesteps = torch.linspace(max_steps, 1, steps, dtype=torch.long).tolist()
        return sorted(set(timesteps), reverse=True)

    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        instrument_ids: torch.Tensor,
        f0_score: torch.Tensor,
        num_denoising_steps: int | None = None,
        use_ddim: bool | None = None,
    ):
        """
        サンプリング

        Args:
            model: UNetDiffusionモデル
            shape: (B, L, D) - 生成したい形状
            instrument_ids: (B,) - 楽器ID
            f0_score: (B, L) - f0スコア
            num_denoising_steps: デノイジング回数（省略時は num_timesteps）
            use_ddim: DDIMを使うか
        """
        batch_size, _, _ = shape
        x = torch.randn(shape, device=self.device)

        max_steps = self.num_timesteps
        steps = num_denoising_steps if num_denoising_steps is not None else max_steps
        if steps < 1 or steps > max_steps:
            raise ValueError(
                f"num_denoising_steps must be between 1 and {max_steps}, got {steps}"
            )

        timesteps = self._build_sampling_timesteps(max_steps, steps)

        pbar: tqdm = tqdm(
            desc="Sampling (DDIM)" if use_ddim else "Sampling",
            total=len(timesteps),
        )

        for idx, i in enumerate(timesteps):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            prev_t = timesteps[idx + 1] if idx + 1 < len(timesteps) else 0
            if use_ddim:
                x = self.denoise_ddim(
                    model, x, t, instrument_ids, f0_score, prev_t=prev_t
                )
            else:
                x = self.denoise(
                    model, x, t, instrument_ids, f0_score, prev_t=prev_t
                )
            pbar.set_postfix({"t": i})
            pbar.update(1)
        pbar.close()
        return x

    def inpaint(
        self,
        model: nn.Module,
        shape: tuple,
        instrument_ids: torch.Tensor,
        f0_score: torch.Tensor,
        mask: torch.Tensor,
        previous_features: torch.Tensor,
        num_denoising_steps: int | None = None,
        use_ddim: bool | None = None,
    ):
        """
        マスクされた範囲のみを再生成するサンプリング。

        Args:
            mask: (B, L) or (B, L, 1)。1の範囲を再生成し、0の範囲は既存featureを順拡散して固定する。
            previous_features: (B, L, D)。再生成前の正規化済みfeature。
        """
        batch_size, _, _ = shape
        if previous_features.shape != shape:
            raise ValueError(
                f"previous_features shape must match shape: {previous_features.shape} != {shape}"
            )

        mask = mask.to(device=self.device, dtype=previous_features.dtype)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        if mask.shape != (batch_size, shape[1], 1):
            raise ValueError(f"mask shape must be (B, L) or (B, L, 1), got {mask.shape}")

        max_steps = self.num_timesteps
        steps = num_denoising_steps if num_denoising_steps is not None else max_steps
        if steps < 1 or steps > max_steps:
            raise ValueError(
                f"num_denoising_steps must be between 1 and {max_steps}, got {steps}"
            )

        timesteps = self._build_sampling_timesteps(max_steps, steps)

        known_x = self._diffuse_or_original(previous_features, timesteps[0])
        x = mask * torch.randn(shape, device=self.device) + (1 - mask) * known_x

        pbar: tqdm = tqdm(
            desc="Inpainting (DDIM)" if use_ddim else "Inpainting",
            total=len(timesteps),
        )

        for idx, i in enumerate(timesteps):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            prev_t = timesteps[idx + 1] if idx + 1 < len(timesteps) else 0
            if use_ddim:
                generated_x = self.denoise_ddim(
                    model, x, t, instrument_ids, f0_score, prev_t=prev_t
                )
            else:
                generated_x = self.denoise(
                    model, x, t, instrument_ids, f0_score, prev_t=prev_t
                )

            known_x = self._diffuse_or_original(previous_features, prev_t)
            x = mask * generated_x + (1 - mask) * known_x
            pbar.set_postfix({"t": i})
            pbar.update(1)
        pbar.close()
        return x

    def _diffuse_or_original(self, x_0: torch.Tensor, t: int) -> torch.Tensor:
        if t == 0:
            return x_0
        t = torch.full((x_0.shape[0],), t, device=self.device, dtype=torch.long)
        return self.add_noise(x_0, t)[0]

    def denoise_with_guidance(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        instrument_ids: torch.Tensor,
        f0_score: torch.Tensor,
        ddsp_decoder: nn.Module,
        loss_fn: nn.Module,
        loss_config,
        split_params_fn: callable,
        observed_mixture: torch.Tensor,
        f0_diff_mean: float,
        f0_diff_std: float,
        f0_original_mean: float,
        f0_original_std: float,
        z_feature_mean: torch.Tensor,
        z_feature_std: torch.Tensor,
        loudness_diff_mean: float,
        loudness_diff_std: float,
        loudness_original_mean: float,
        loudness_original_std: float,
        loudness_score: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        instrument_names: list[int | Instrument] | None = None,
        segment_num: int | None = None,
        num_instruments: int | None = None,
        pbar: tqdm | None = None,
        output_dir: str | None = None,
        sampling_rate: int | None = None,
        direct_optim: list[str] | None = None,
        guiding_params: list[str] = ["pitch", "loudness", "z_feature"],
    ) -> torch.Tensor:
        """
        データ整合性付きノイズ除去（guided sampling）

        Args:
            model: UNetDiffusionモデル
            x: (B, W, H) - ノイズが加えられた合成パラメータ
                B = num_instruments * segment_num（各サンプルが楽器×セグメントを表す）
            t: (B,) - 時間ステップ
            instrument_ids: (B,) - 楽器ID（各サンプルが楽器×セグメントを表す）
            f0_score: (B, L) - f0スコア
            ddsp_model: DDSPモデル
            loss_fn: ロス関数（Lossクラス）
            loss_config: LossConfig - ロス設定
            split_params_fn: パラメータ分割関数（split_params）
            observed_mixture: (1, T) - 観測された混合音（全楽器の合計、全セグメント結合済み）
            f0_diff_mean, f0_diff_std: f0_diffの統計情報
            f0_original_mean, f0_original_std: 元のf0の統計情報
            z_feature_mean, z_feature_std: z_featureの統計情報
            loudness_diff_mean, loudness_diff_std: loudness_diffの統計情報
            loudness_original_mean, loudness_original_std: 元のloudnessの統計情報
            loudness_score: (B, L) - loudness_score
            guidance_scale: ガイダンススケール（勾配のスケーリング係数）
            cfg_guidance_scale: CFGのガイダンススケール（1.0でCFG無効、>1.0で条件の影響を強化）
            instrument_names: 楽器名（TDLossやFDLossを使用する場合に必要）
            segment_num: セグメント数（必須）
            num_instruments: 楽器数（必須）
            pbar: tqdm | None = None - プログレスバー
            output_dir: str | None = None - 出力ディレクトリ
            sampling_rate: int | None = None - サンプリングレート
            direct_optim: list[str] | None = None # ["pitch", "loudness", "z_feature"]
        Returns:
            x_{t-1}: (B, L, D) - 次のステップのサンプル
        """
        if direct_optim is not None:
            self.optimizer.zero_grad()
        if segment_num is None or num_instruments is None:
            raise ValueError("segment_num and num_instruments must be provided")
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1
        alpha = self.alphas[t_idx]  # (N,)
        alpha_bar = self.alpha_bars[t_idx]  # (N,)
        beta = self.betas[t_idx]  # (N,)

        # t=1の場合は前のステップがないので、特別処理
        mask = (t > 1).float()
        alpha_bar_prev = torch.where(
            t > 1, self.alpha_bars[t_idx - 1], torch.ones_like(alpha_bar)
        )  # (N,)

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1)
        beta = beta.view(N, 1, 1)

        with torch.no_grad():
            eps_pred = model(x, t, instrument_ids, f0_score)

        # x_tから勾配を取るために、requires_grad=Trueにする
        x_grad = x.detach().requires_grad_(True)

        # 予測されたノイズから x_0 を推定（勾配計算用）
        pred_x_0 = (x_grad - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)

        # 全バッチを一度に処理
        all_f0s, _, all_loudnesses, _, all_z_features = split_params_fn(
            aggregated=pred_x_0,  # (B, L, 18)
            f0_diff_mean=f0_diff_mean,
            f0_diff_std=f0_diff_std,
            f0_original_mean=f0_original_mean,
            f0_original_std=f0_original_std,
            loudness_diff_mean=loudness_diff_mean,
            loudness_diff_std=loudness_diff_std,
            loudness_original_mean=loudness_original_mean,
            loudness_original_std=loudness_original_std,
            z_feature_mean=z_feature_mean,
            z_feature_std=z_feature_std,
            f0_score=f0_score,  # (B, L) - normalized
            loudness_score=loudness_score,  # (B, L) - raw loudness score
            direct_optim=direct_optim,
            pitches=self.pitches if direct_optim is not None and "pitch" in direct_optim else None,
            loudnesses=self.loudnesses if direct_optim is not None and "loudness" in direct_optim else None,
            z_features=self.z_features if direct_optim is not None and "z_feature" in direct_optim else None,
        )
        # all_f0s: (B, L), all_loudnesses: (B, L), all_z_features: (B, L, 16)
        all_f0s = all_f0s.float()  # float32に変換
        all_loudnesses = all_loudnesses.float()
        all_z_features = all_z_features.float()

        # DDSP decoder で全バッチの音声を一度に生成
        ddsp_decoder.train()  # 勾配計算のためにtrainモード
        for p in ddsp_decoder.parameters():
            p.requires_grad_(False)

        f0_input = all_f0s.unsqueeze(-1).float()  # (B, L, 1)
        loudness_input = all_loudnesses.unsqueeze(-1)  # (B, L, 1)
        z_feature_input = all_z_features  # (B, L, 16)

        signals, _, _, _ = ddsp_decoder(
            f0_input,
            loudness_input,
            z_feature_input,
        )
        signals: torch.Tensor = signals.squeeze(-1)  # (B, T)

        # ロス計算用に楽器ごとにセグメントを結合（ベクトル化）
        # signals: (B, T_seg) where B = num_instruments * segment_num
        # -> (num_instruments, segment_num, T_seg) -> (num_instruments, T_total)
        signals_reshaped = signals.view(num_instruments, segment_num, -1)  # (num_instruments, segment_num, T_seg)
        instrument_signals = signals_reshaped.reshape(num_instruments, -1)  # (num_instruments, T_total)
        instrument_signals = instrument_signals.unsqueeze(1)  # (num_instruments, 1, T_total)
        
        # 全楽器の混合音を作成
        mixed_signal = instrument_signals.sum(dim=0, keepdim=True)  # (1, 1, T_total)
        mixed_signal = mixed_signal.squeeze(0)  # (1, T_total)

        current_t = int(t[0].item()) if t.numel() > 0 else 0
        
        # パラメータも楽器ごとにセグメントを結合（ベクトル化）
        # all_f0s: (B, L_seg) -> (num_instruments, segment_num, L_seg) -> (num_instruments, L_total)
        pitches_reshaped = all_f0s.view(num_instruments, segment_num, -1)  # (num_instruments, segment_num, L_seg)
        pitches_stacked = pitches_reshaped.reshape(num_instruments, -1)  # (num_instruments, L_total)
        
        loudnesses_reshaped = all_loudnesses.view(num_instruments, segment_num, -1)  # (num_instruments, segment_num, L_seg)
        loudnesses_stacked = loudnesses_reshaped.reshape(num_instruments, -1)  # (num_instruments, L_total)
        
        z_features_reshaped = all_z_features.view(num_instruments, segment_num, -1, 16)  # (num_instruments, segment_num, L_seg, 16)
        z_features_stacked = z_features_reshaped.reshape(num_instruments, -1, 16)  # (num_instruments, L_total, 16)

        # 観測混合音とのロスを計算
        # observed_mixtureとmixed_signalの長さを揃える
        min_length = min(mixed_signal.shape[1], observed_mixture.shape[1])
        mixed_signal_trunc = mixed_signal[:, :min_length]
        observed_mixture_trunc = observed_mixture[:, :min_length]

        # Lossクラスを使用してロスを計算
        # LossInputsを動的にインポート（循環インポートを避けるため）
        from api.models.loss import LossInputs
        
        if instrument_names is None:
            raise ValueError("instrument_names is required")
        
        # LossInputsを作成
        loss_inputs = LossInputs.from_results(
            loss_config=loss_config,
            signal_pred=mixed_signal_trunc.reshape(-1),
            signal_target=observed_mixture_trunc.reshape(-1),
            loudness=loudnesses_stacked,
            pitch=pitches_stacked,
            z_feature=z_features_stacked,
            instrument_names=instrument_names,
        )
        
        # Lossクラスを使用してロスを計算
        loss = loss_fn(loss_inputs)

        # ロスの勾配を計算（x_tに対して）
        # 勾配は自動的に各バッチ要素（楽器×セグメント）にマッピングされる
        grad = torch.autograd.grad(loss, x_grad, retain_graph=True, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(x_grad)

        # 勾配の正規化
        eps = 1e-8
        norm_dims = tuple(range(grad.ndim - 1))
        grad_norm_per_dim = torch.norm(grad, dim=norm_dims, keepdim=True)
        grad_normalized = grad / (grad_norm_per_dim + eps)

        guidance_term = guidance_scale * grad_normalized

        if pbar is not None:
            current_t = t[0].item() if len(t) > 0 else 0
            pbar.set_postfix({"t": current_t, "loss": f"{loss.item():.4f}", "guidance_scale": f"{guidance_scale:.3f}"})
            pbar.update(1)

        if "pitch" in guiding_params:
            x[:, :, 0] = x[:, :, 0] - guidance_term[:, :, 0]
        if "loudness" in guiding_params:
            x[:, :, 1] = x[:, :, 1] - guidance_term[:, :, 1]
        if "z_feature" in guiding_params:
            x[:, :, 2:] = x[:, :, 2:] - guidance_term[:, :, 2:]

        # 修正されたscoreから x_0 を再推定
        pred_x_0_guided: torch.Tensor = (x - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)

        # 前のステップへの遷移
        mu: torch.Tensor = (torch.sqrt(alpha_bar_prev) * beta / (1 - alpha_bar)) * pred_x_0_guided + (
            torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)
        ) * x

        noise = torch.randn_like(x, device=self.device)
        noise = noise * mask.view(N, 1, 1)  # t=1の場合はノイズを追加しない

        std = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        std = std.view(N, 1, 1)


        if direct_optim is not None:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            x = mu + noise * std
            if "pitch" in direct_optim:
                pitches_diff = self.pitches.reshape(-1, f0_score.shape[1]) - denormalize(f0_score, f0_original_mean, f0_original_std)
                x[:, :, 0] = normalize(pitches_diff, f0_diff_mean, f0_diff_std)
            if "loudness" in direct_optim:
                loudnesses_diff = denormalize(self.loudnesses, loudness_original_mean, loudness_original_std).reshape(-1, loudness_score.shape[1]) - loudness_score
                x[:, :, 1] = normalize(loudnesses_diff, loudness_diff_mean, loudness_diff_std)
            if "z_feature" in direct_optim:
                z_features_reshaped = self.z_features.reshape(-1, self.z_features.shape[2], 16)
                x[:, :, 2:] = normalize(z_features_reshaped, z_feature_mean, z_feature_std)
            return x
        
        return mu + noise * std

    def sample_with_guidance(
        self,
        model: nn.Module,
        shape: tuple,
        instrument_ids: torch.Tensor,
        f0_score: torch.Tensor,
        ddsp_decoder: nn.Module,
        loss_fn: nn.Module,
        loss_config,
        split_params_fn: callable,
        observed_mixture: torch.Tensor,
        f0_diff_mean: float,
        f0_diff_std: float,
        f0_original_mean: float,
        f0_original_std: float,
        z_feature_mean: torch.Tensor,
        z_feature_std: torch.Tensor,
        loudness_diff_mean: float,
        loudness_diff_std: float,
        loudness_original_mean: float,
        loudness_original_std: float,
        loudness_score: torch.Tensor | None = None,
        guidance_scale_start: float = 1.0,
        guidance_scale_end: float = 1.0,
        instrument_names: list[int | Instrument] | None = None,
        segment_num: int | None = None,
        num_instruments: int | None = None,
        output_dir: str | None = None,
        sampling_rate: int | None = None,
        # for direct optimization
        pitches: torch.Tensor | None = None,
        loudnesses: torch.Tensor | None = None,
        z_features: torch.Tensor | None = None,
        lr: float = 0.1,
        direct_optim: list[str] | None = None,
        guiding_params: list[str] = ["pitch", "loudness", "z_feature"],
    ):
        """
        ガイダンス付きサンプリング（guided sampling with data consistency）

        Args:
            model: UNetDiffusionモデル
            shape: (B, L, D) - 生成したい形状
                B = num_instruments * segment_num（各サンプルが楽器×セグメントを表す）
            instrument_ids: (B,) - 楽器ID
            f0_score: (B, L) - f0スコア
            ddsp_model: DDSPモデル
            loss_fn: ロス関数（Lossクラス）
            loss_config: LossConfig - ロス設定
            split_params_fn: パラメータ分割関数（split_params）
            observed_mixture: (1, T) - 観測された混合音（全楽器の合計、全セグメント結合済み）
            f0_diff_mean, f0_diff_std: f0_diffの統計情報
            f0_original_mean, f0_original_std: 元のf0の統計情報
            z_feature_mean, z_feature_std: z_featureの統計情報
            loudness_diff_mean, loudness_diff_std: loudness_diffの統計情報
            loudness_original_mean, loudness_original_std: 元のloudnessの統計情報
            loudness_score: (B, L) - loudness_score
            guidance_scale_start: ガイダンススケールの開始値（最初のtimestepで使用）
            guidance_scale_end: ガイダンススケールの終了値（最後のtimestepで使用）
            instrument_names: 楽器名（TDLossやFDLossを使用する場合に必要）
            segment_num: セグメント数（Noneの場合は従来の動作）
            num_instruments: 楽器数（Noneの場合は従来の動作）
            output_dir: 出力ディレクトリ（音声ファイルを保存する場合）
            sampling_rate: サンプリングレート（音声ファイルを保存する場合）
            direct_optim: list[str] | None = None # ["pitch", "loudness", "z_feature"]

        Returns:
            x_0: (B, L, D) - 生成された合成パラメータ
        """
        batch_size, _,  _= shape
        x = torch.randn(shape, device=self.device)

        if direct_optim is not None:
            if "pitch" in direct_optim:
                pitches_reshaped = pitches.reshape(-1, f0_score.shape[1])
                pitches_diff = pitches_reshaped - denormalize(f0_score, f0_original_mean, f0_original_std)
                x[:, :, 0] = normalize(pitches_diff, f0_diff_mean, f0_diff_std)
            if "loudness" in direct_optim:  
                loudnesses_reshaped = denormalize(loudnesses, loudness_original_mean, loudness_original_std).reshape(-1, loudness_score.shape[1])
                loudnesses_diff = loudnesses_reshaped - loudness_score
                x[:, :, 1] = normalize(loudnesses_diff, loudness_diff_mean, loudness_diff_std)
            if "z_feature" in direct_optim:
                z_features_reshaped = z_features.reshape(-1, z_features.shape[2], 16)
                x[:, :, 2:] = normalize(z_features_reshaped, z_feature_mean, z_feature_std)
        
        pbar: tqdm = tqdm(desc="Sampling with guidance", total=self.num_timesteps)

        if direct_optim is not None:
            params_list = []
            if "pitch" in direct_optim:
                self.pitches = pitches.detach().requires_grad_(True)
                params_list.append(self.pitches)
            if "loudness" in direct_optim:
                self.loudnesses = loudnesses.detach().requires_grad_(True)
                params_list.append(self.loudnesses)
            if "z_feature" in direct_optim:
                self.z_features = z_features.detach().requires_grad_(True)
                params_list.append(self.z_features)
            self.optimizer = torch.optim.Adam(
                params_list, lr=lr
            )
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[2000, 3000], gamma=0.1
            )


        for i in range(self.num_timesteps, 0, -1):
            progress = (self.num_timesteps - i) / max(self.num_timesteps - 1, 1)
            gamma = 4.0
            progress_nl = progress ** gamma
            current_guidance_scale = (
                guidance_scale_start * (1 - progress_nl) + guidance_scale_end * progress_nl
            )
            
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)   
            x = self.denoise_with_guidance(
                model=model,
                x=x,
                t=t,
                instrument_ids=instrument_ids,
                f0_score=f0_score,
                ddsp_decoder=ddsp_decoder,
                loss_fn=loss_fn,
                loss_config=loss_config,
                split_params_fn=split_params_fn,
                observed_mixture=observed_mixture,
                f0_diff_mean=f0_diff_mean,
                f0_diff_std=f0_diff_std,
                f0_original_mean=f0_original_mean,
                f0_original_std=f0_original_std,
                z_feature_mean=z_feature_mean,
                z_feature_std=z_feature_std,
                loudness_diff_mean=loudness_diff_mean,
                loudness_diff_std=loudness_diff_std,
                loudness_original_mean=loudness_original_mean,
                loudness_original_std=loudness_original_std,
                loudness_score=loudness_score,
                guidance_scale=current_guidance_scale,
                instrument_names=instrument_names,
                segment_num=segment_num,
                num_instruments=num_instruments,
                pbar=pbar,
                output_dir=output_dir,
                sampling_rate=sampling_rate,
                direct_optim=direct_optim,
                guiding_params=guiding_params,
            )

        pbar.close()
        return x

def get_diffusion_model(
    time_embed_dim: int = 128,
    num_instruments: int = 10,
    film_cond_dim: int = 64,
    cfg_dropout_prob: float = 0.1,
    f0_score_scale: float = 1.0,
    # UNet parameters
    down1_out_ch: int = 64,
    down2_out_ch: int = 128,
    down3_out_ch: int = 256,
    bot1_out_ch: int = 512,
) -> nn.Module:
    """
    Factory function to create diffusion model based on config
    
    Args:
        model_type: "unet" or "dit"
        cfg_dropout_prob: CFG用のdropout確率（学習時に楽器ラベルを無効化する確率、0.1が推奨）
        その他のパラメータは各モデルの設定
    """
    return UNetDiffusion(
        time_embed_dim=time_embed_dim,
        num_instruments=num_instruments,
        film_cond_dim=film_cond_dim,
        down1_out_ch=down1_out_ch,
        down2_out_ch=down2_out_ch,
        down3_out_ch=down3_out_ch,
        bot1_out_ch=bot1_out_ch,
        cfg_dropout_prob=cfg_dropout_prob,
        f0_score_scale=f0_score_scale,
    )
