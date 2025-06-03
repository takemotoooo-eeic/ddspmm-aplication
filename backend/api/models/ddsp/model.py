import torch
import torch.nn as nn
from .core import mlp, gru, scale_function, remove_above_nyquist, upsample
from .core import harmonic_synth, amp_to_impulse_response, fft_convolve
import math
import torchaudio


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
