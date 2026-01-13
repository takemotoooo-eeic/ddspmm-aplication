from __future__ import annotations

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .instrument import Instrument


from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler
import soundfile as sf

from api.models.diffusion.pitch import cent_to_hz


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
        use_note_mask: bool = False,
        note_mask_scale: float = 1.0,
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

        # f0_scoreエンコーダー（2次元画像として扱う: (B, 1, 18, L)）
        # f0_scoreは時間軸方向にブロードキャストして、高さ方向に複製
        self.f0_score_mlp = nn.Sequential(
            nn.Conv2d(1, in_ch, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
        )

        # note_maskエンコーダー（f0_scoreと同様）
        self.use_note_mask = use_note_mask
        if use_note_mask:
            self.note_mask_mlp = nn.Sequential(
                nn.Conv2d(1, in_ch, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_ch, in_ch, kernel_size=1),
            )

        if film_cond_dim is not None:
            self.film = FiLM(film_cond_dim, out_ch)
        else:
            self.film = None
        
        self.f0_score_scale = f0_score_scale
        self.note_mask_scale = note_mask_scale

    def forward(
        self,
        x: torch.Tensor,
        time_embed: torch.Tensor,
        film_cond: Optional[torch.Tensor] = None,
        f0_score: Optional[torch.Tensor] = None,
        note_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: (B, C, H, W) - 2次元画像 (H=18特徴次元, W=L時間軸)
            time_embed: (B, time_embed_dim)
            film_cond: (B, film_cond_dim) or None
            f0_score: (B, W) or None - f0スコア（時間変動を保持、Wは時間軸の長さ）
            note_mask: (B, W) or None - 音符存在マスク（時間変動を保持、Wは時間軸の長さ）
        """
        B, C, H, W = x.shape

        # Time embedding
        v = self.time_mlp(time_embed)  # (B, in_ch)
        v = v.view(B, C, 1, 1)  # (B, in_ch, 1, 1)
        v = v.expand(B, C, H, W)  # (B, in_ch, H, W) - ブロードキャスト

        # f0_scoreエンコーディング（時間変動を保持）
        # f0_score (B, W) -> (B, 1, 1, W) -> (B, 1, H, W) にブロードキャスト
        if f0_score is not None:
            f0_score_input = f0_score.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, W)
            f0_score_input = f0_score_input.expand(B, 1, H, W)  # (B, 1, H, W)
            f0_score_encoded = self.f0_score_mlp(f0_score_input)  # (B, in_ch, H, W)
            v = v + self.f0_score_scale * f0_score_encoded  # (B, in_ch, H, W) - スケーリング適用

        # note_maskエンコーディング（時間変動を保持）
        if self.use_note_mask and note_mask is not None:
            note_mask_input = note_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, W)
            note_mask_input = note_mask_input.expand(B, 1, H, W)  # (B, 1, H, W)
            note_mask_encoded = self.note_mask_mlp(note_mask_input)  # (B, in_ch, H, W)
            v = v + self.note_mask_scale * note_mask_encoded  # (B, in_ch, H, W) - スケーリング適用

        x = x + v

        # Convolution
        y = self.convs(x)

        # FiLM conditioning
        if self.film is not None and film_cond is not None:
            y = self.film(y, film_cond)

        return y


class DiTBlock(nn.Module):
    """Diffusion Transformer Block with FiLM and/or Cross-Attention conditioning"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 128,
        film_cond_dim: int | None = None,
        use_film: bool = False,
        use_cross_attention: bool = False,
        f0_score_scale: float = 1.0,
        use_note_mask: bool = False,
        note_mask_scale: float = 1.0,
    ):
        super().__init__()
        self.use_film = use_film
        self.use_cross_attention = use_cross_attention
        
        # Layer norm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        if use_cross_attention:
            self.norm_cross = nn.LayerNorm(hidden_dim)
        
        # Self-attention
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Cross-attention (for instrument conditioning)
        if use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            if film_cond_dim is None:
                raise ValueError("film_cond_dim must be specified when use_cross_attention is True")
            self.instrument_proj = nn.Linear(film_cond_dim, hidden_dim)
        
        # MLP
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # f0_scoreエンコーディング（ConvBlockと同様にadditive conditioning）
        # f0_score (B, L) -> (B, L, 1) -> (B, L, hidden_dim)
        self.f0_score_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # note_maskエンコーディング（f0_scoreと同様）
        self.use_note_mask = use_note_mask
        if use_note_mask:
            self.note_mask_mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        
        # FiLM conditioning
        if use_film:
            if film_cond_dim is None:
                raise ValueError("film_cond_dim must be specified when use_film is True")
            self.film = FiLM(film_cond_dim, hidden_dim)
        
        self.f0_score_scale = f0_score_scale
        self.note_mask_scale = note_mask_scale
    
    def forward(
        self,
        x: torch.Tensor,
        time_embed: torch.Tensor,
        film_cond: torch.Tensor | None = None,
        f0_score: torch.Tensor | None = None,
        note_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            x: (B, L, hidden_dim) - 入力シーケンス
            time_embed: (B, time_embed_dim) - Time embedding
            film_cond: (B, film_cond_dim) or None - Instrument embedding for FiLM
            f0_score: (B, L) or None - f0_score for additive conditioning
            note_mask: (B, L) or None - note_mask for additive conditioning
        """
        B, L, D = x.shape
        
        # Time embeddingを加算
        time_emb = self.time_mlp(time_embed)  # (B, hidden_dim)
        v = time_emb.unsqueeze(1)  # (B, 1, hidden_dim) -> broadcast to (B, L, hidden_dim)
        
        # f0_scoreエンコーディング（ConvBlockと同様にadditive conditioning）
        # f0_score (B, L) -> (B, L, 1) -> (B, L, hidden_dim)
        if f0_score is not None:
            f0_score_input = f0_score.unsqueeze(-1)  # (B, L, 1)
            f0_score_encoded = self.f0_score_mlp(f0_score_input)  # (B, L, hidden_dim)
            v = v + self.f0_score_scale * f0_score_encoded  # (B, L, hidden_dim) - スケーリング適用
        
        # note_maskエンコーディング（時間変動を保持）
        if self.use_note_mask and note_mask is not None:
            note_mask_input = note_mask.unsqueeze(-1)  # (B, L, 1)
            note_mask_encoded = self.note_mask_mlp(note_mask_input)  # (B, L, hidden_dim)
            v = v + self.note_mask_scale * note_mask_encoded  # (B, L, hidden_dim) - スケーリング適用
        
        x = x + v  # Time embedding + f0_score + note_maskを加算
        
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # (B, L, hidden_dim)
        x = x + attn_out
        
        # Cross-attention (for instrument conditioning)
        if self.use_cross_attention:
            x_norm = self.norm_cross(x)
            
            # Cross-attention for instrument
            if film_cond is not None:
                instrument_emb = self.instrument_proj(film_cond)  # (B, hidden_dim)
                # instrument_embをクエリとして、xをkey/valueとして使用
                instrument_query = instrument_emb.unsqueeze(1)  # (B, 1, hidden_dim)
                cross_attn_out, _ = self.cross_attn(instrument_query, x_norm, x_norm)  # (B, 1, hidden_dim)
                # ブロードキャストして加算
                x = x + cross_attn_out.expand(B, L, D)
        
        # MLP
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        # FiLM conditioning
        if self.use_film and film_cond is not None:
            x = self.film(x, film_cond)  # x: (B, L, hidden_dim) -> FiLM適用
        
        return x


class DiTDiffusion(nn.Module):
    """Diffusion Transformer for parameter generation"""
    
    def __init__(
        self,
        in_dim: int = 18,
        time_embed_dim: int = 128,
        num_instruments: int = 10,
        film_cond_dim: int = 64,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_film: bool = True,
        use_cross_attention: bool = False,
        cfg_dropout_prob: float = 0.1,
        f0_score_scale: float = 1.0,
        use_note_mask: bool = False,
        note_mask_scale: float = 1.0,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.cfg_dropout_prob = cfg_dropout_prob
        
        # Instrument embedding for FiLM/Cross-attention
        self.instrument_emb = nn.Embedding(num_instruments, film_cond_dim)
        
        # Null embedding for CFG (Classifier-Free Guidance)
        # 学習可能なnull embedding（条件なしの場合に使用）
        self.null_instrument_emb = nn.Parameter(torch.zeros(1, film_cond_dim))
        
        # Input projection (patch embedding)
        # Each timestep is treated as a token
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Positional encoding (learnable)
        # Max sequence length is set to 4096 for flexibility
        max_seq_len = 4096
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                time_embed_dim=time_embed_dim,
                film_cond_dim=film_cond_dim,
                use_film=use_film,
                use_cross_attention=use_cross_attention,
                f0_score_scale=f0_score_scale,
                use_note_mask=use_note_mask,
                note_mask_scale=note_mask_scale,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, in_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        instrument_ids: torch.Tensor,
        f0_score: torch.Tensor,
        instrument_cond_mask: torch.Tensor | None = None,
        note_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            x: (B, L, 18) - ノイズが加えられた合成パラメータ
            timesteps: (B,) - 時間ステップ
            instrument_ids: (B,) - 楽器ID
            f0_score: (B, L) - f0スコア（conditioning用）
            instrument_cond_mask: (B,) or None - CFG用の条件マスク（True=条件付き、False=条件なし）
                                     Noneの場合は学習時にランダムにドロップアウト
            note_mask: (B, L) or None - 音符存在マスク（conditioning用）
        """
        B, L, D = x.shape
        
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
        
        # Input projection
        x = self.input_proj(x)  # (B, L, hidden_dim)
        
        # Add positional encoding
        if L <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, :L, :]
        else:
            # Sequence is longer than max_seq_len, interpolate positional encoding
            pos_emb = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=L,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            x = x + pos_emb
        
        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x, t, film_cond, f0_score, note_mask)
        
        # Final norm and output projection
        x = self.final_norm(x)
        x = self.output_proj(x)  # (B, L, 18)
        
        return x


class DiffusersUNet2DModelWrapper(nn.Module):
    """diffusersのUNet2DModelを使ったラッパークラス（既存のUNetDiffusionと同じインターフェース）"""
    
    def __init__(
        self,
        in_dim: int = 18,  # f0(1) + loudness(1) + z_feature(16)
        time_embed_dim: int = 128,
        num_instruments: int = 10,
        film_cond_dim: int = 64,
        cfg_dropout_prob: float = 0.1,
        f0_score_scale: float = 1.0,
        use_note_mask: bool = False,
        note_mask_scale: float = 1.0,
        # UNet2DModelのパラメータ
        sample_size: int = 18,  # 高さ（特徴次元）
        sample_width: int | None = None,  # 横幅（時間軸の長さ、8の倍数にパディング後、Noneの場合は動的）
        in_channels: int = 1,  # 入力チャネル数（通常は1、f0_score/note_maskをconcatする場合は増やす）
        out_channels: int = 1,  # 出力チャネル数
        down_block_types: tuple = ("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types: tuple = ("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        mid_block_type: str | None = "UNetMidBlock2D",  # mid blockのタイプ（Noneで無効化）
        block_out_channels: tuple = (64, 128, 256),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        attention_head_dim: int = 8,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
    ):
        super().__init__()        
        self.in_dim = in_dim
        self.time_embed_dim = time_embed_dim
        self.film_cond_dim = film_cond_dim
        self.cfg_dropout_prob = cfg_dropout_prob
        self.f0_score_scale = f0_score_scale
        self.use_note_mask = use_note_mask
        self.note_mask_scale = note_mask_scale
        self.sample_width = sample_width  # 横幅（固定長、Noneの場合は動的）
        
        # f0_scoreとnote_maskを入力チャネルにconcat（楽器IDはclass_labelsとして条件付け）
        extra_channels = 0
        if True:  # f0_scoreは常に使用
            extra_channels += 1
        if use_note_mask:
            extra_channels += 1
        
        self.num_instruments = num_instruments
        
        # UNet2DModelの初期化
        # sample_sizeは (height, width) のタプルとして指定
        # height = 24（特徴次元18を24にパディング）、width = sample_width（時間軸の長さ、8の倍数にパディング後）
        # 高さは24に固定（forward内でパディングするため）
        target_height = 24
        if sample_width is not None:
            # 固定サイズを指定（(height, width)のタプル）
            sample_size_tuple = (target_height, sample_width)
        else:
            # 動的サイズ（sample_size=None）
            sample_size_tuple = None
        
        # UNet2DModelの初期化時にclass_embeddingを設定
        # class_embed_type=None, num_class_embeds=num_instruments で通常のembeddingを使用
        try:
            if sample_size_tuple is not None:
                # 固定サイズ (height, width) を指定
                self.unet = UNet2DModel(
                    sample_size=sample_size_tuple,  # (24, sample_width)のタプル
                    in_channels=in_channels + extra_channels,  # 元の入力 + f0_score + note_mask
                    out_channels=out_channels,
                    resnet_time_scale_shift="scale_shift",
                    down_block_types=down_block_types,
                    up_block_types=up_block_types,
                    mid_block_type=mid_block_type,  # mid blockのタイプ
                    block_out_channels=block_out_channels,
                    layers_per_block=layers_per_block,
                    act_fn=act_fn,
                    attention_head_dim=attention_head_dim,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    class_embed_type=None,  # class_labelsを使用
                    num_class_embeds=num_instruments,  # 楽器数
                )
            else:
                # 動的サイズ（sample_size=None）
                self.unet = UNet2DModel(
                    sample_size=None,  # 動的サイズを許可（可変長Lに対応）
                    in_channels=in_channels + extra_channels,  # 元の入力 + f0_score + note_mask
                    out_channels=out_channels,
                    resnet_time_scale_shift="scale_shift",
                    down_block_types=down_block_types,
                    up_block_types=up_block_types,
                    mid_block_type=mid_block_type,  # mid blockのタイプ
                    block_out_channels=block_out_channels,
                    layers_per_block=layers_per_block,
                    act_fn=act_fn,
                    attention_head_dim=attention_head_dim,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    class_embed_type=None,  # class_labelsを使用
                    num_class_embeds=num_instruments,  # 楽器数
                )
        except (TypeError, ValueError) as e:
            # sample_sizeの指定方法がサポートされていない場合は、固定値（高さのみ）を使用
            # 実際の入力サイズはforward内でパディングによって調整される
            self.unet = UNet2DModel(
                sample_size=target_height,  # 固定値（高さ24）
                in_channels=in_channels + extra_channels,  # 元の入力 + f0_score + note_mask
                resnet_time_scale_shift="scale_shift",
                out_channels=out_channels,
                down_block_types=down_block_types,
                up_block_types=up_block_types,
                mid_block_type=mid_block_type,  # mid blockのタイプ
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                act_fn=act_fn,
                attention_head_dim=attention_head_dim,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                class_embed_type=None,  # class_labelsを使用
                num_class_embeds=num_instruments,  # 楽器数
            )
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        instrument_ids: torch.Tensor,
        f0_score: torch.Tensor,
        instrument_cond_mask: torch.Tensor | None = None,
        note_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            x: (B, L, 18) - ノイズが加えられた合成パラメータ
            timesteps: (B,) - 時間ステップ
            instrument_ids: (B,) - 楽器ID
            f0_score: (B, L) - f0スコア（conditioning用）
            instrument_cond_mask: (B,) or None - CFG用の条件マスク（True=条件付き、False=条件なし）
            note_mask: (B, L) or None - 音符存在マスク（conditioning用）
        """
        B, L, D = x.shape
        assert D == self.in_dim, f"Expected in_dim={self.in_dim}, but got D={D}"
        
        # パディングを適用して、UNet2DModelが処理できるサイズにする
        # sample_widthが指定されている場合はその値に合わせてパディング
        # 指定されていない場合は、8の倍数にパディング（3回ダウンサンプリングする場合、8=2^3の倍数が必要）
        original_L = L
        if self.sample_width is not None:
            # 固定サイズにパディング
            if L < self.sample_width:
                pad_length = self.sample_width - L
                # 最後の値を複製してパディング
                x = torch.cat([x, x[:, -1:, :].expand(-1, pad_length, -1)], dim=1)
                L = x.shape[1]
                # f0_scoreも同様にパディング
                f0_score = torch.cat([f0_score, f0_score[:, -1:].expand(-1, pad_length)], dim=1)
                # note_maskも同様にパディング
                if note_mask is not None:
                    note_mask = torch.cat([note_mask, note_mask[:, -1:].expand(-1, pad_length)], dim=1)
            elif L > self.sample_width:
                # データがsample_widthより長い場合は切り詰め（通常は発生しないはず）
                x = x[:, :self.sample_width, :]
                L = self.sample_width
                f0_score = f0_score[:, :self.sample_width]
                if note_mask is not None:
                    note_mask = note_mask[:, :self.sample_width]
        else:
            # 動的サイズの場合、8の倍数にパディング
            if L % 8 != 0:
                pad_length = 8 - (L % 8)
                # 最後の値を複製してパディング
                x = torch.cat([x, x[:, -1:, :].expand(-1, pad_length, -1)], dim=1)
                L = x.shape[1]
                # f0_scoreも同様にパディング
                f0_score = torch.cat([f0_score, f0_score[:, -1:].expand(-1, pad_length)], dim=1)
                # note_maskも同様にパディング
                if note_mask is not None:
                    note_mask = torch.cat([note_mask, note_mask[:, -1:].expand(-1, pad_length)], dim=1)
        
        # (B, L, 18) -> (B, 18, L) -> (B, 1, 18, L) - 2次元画像として扱う
        x = x.transpose(1, 2)  # (B, 18, L)
        original_H = x.shape[1]  # 元の高さ（18）
        
        # 高さを24にパディング（UNet2DModelのダウンサンプリング/アップサンプリングに対応）
        target_height = 24
        if original_H < target_height:
            pad_height = target_height - original_H
            # 最後の行を複製してパディング
            x = torch.cat([x, x[:, -1:, :].expand(-1, pad_height, -1)], dim=1)  # (B, 24, L)
        elif original_H > target_height:
            # 高さが24より大きい場合は切り詰め（通常は発生しないはず）
            x = x[:, :target_height, :]
        
        x = x.unsqueeze(1)  # (B, 1, 24, L)
        H = x.shape[2]  # パディング後の高さ（24）
        
        # f0_scoreを入力チャネルにconcat
        # f0_score: (B, L) -> (B, 1, 1, L) -> expand -> (B, 1, 24, L)
        if f0_score is not None:
            f0_score_2d = f0_score.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
            f0_score_2d = f0_score_2d.expand(B, 1, H, L)  # (B, 1, 24, L)
            f0_score_2d = f0_score_2d * self.f0_score_scale
            x = torch.cat([x, f0_score_2d], dim=1)  # (B, 1 + 1, 24, L)
        
        # note_maskを入力チャネルにconcat
        if self.use_note_mask and note_mask is not None:
            note_mask_2d = note_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
            note_mask_2d = note_mask_2d.expand(B, 1, H, L)  # (B, 1, 24, L)
            note_mask_2d = note_mask_2d * self.note_mask_scale
            x = torch.cat([x, note_mask_2d], dim=1)  # (B, 1 + 1 or 2, 24, L)
        
        class_labels = instrument_ids
        
        noise_pred = self.unet(
            sample=x,  # (B, C, 24, L)
            timestep=timesteps,  # (B,)
            class_labels=class_labels,
            return_dict=False,
        )[0]  # (B, out_channels, 24, L)
        
        noise_pred = noise_pred.squeeze(1)  # (B, 24, L)
        if noise_pred.shape[1] > original_H:
            noise_pred = noise_pred[:, :original_H, :]  # (B, 18, L)
        noise_pred = noise_pred.transpose(1, 2)  # (B, L, 18)
        
        if noise_pred.shape[1] > original_L:
            noise_pred = noise_pred[:, :original_L, :]
        
        return noise_pred
    


class UNetDiffusion(nn.Module):
    """合成パラメータ生成用のUNetベースのDiffusionモデル（2次元画像版）"""

    def __init__(
        self,
        in_dim: int = 18,  # f0(1) + loudness(1) + z_feature(16)
        time_embed_dim: int = 128,
        num_instruments: int = 10,
        film_cond_dim: int = 64,
        down1_out_ch: int = 64,
        down2_out_ch: int = 128,
        down3_out_ch: int = 256,
        bot1_out_ch: int = 512,
        cfg_dropout_prob: float = 0.1,
        f0_score_scale: float = 1.0,
        use_note_mask: bool = False,
        note_mask_scale: float = 1.0,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.in_dim = in_dim
        self.cfg_dropout_prob = cfg_dropout_prob

        # Instrument embedding for FiLM
        self.instrument_emb = nn.Embedding(num_instruments, film_cond_dim)
        
        # Null embedding for CFG (Classifier-Free Guidance)
        # 学習可能なnull embedding（条件なしの場合に使用）
        self.null_instrument_emb = nn.Parameter(torch.zeros(1, film_cond_dim))

        # 入力チャネル数は1（2次元画像として扱う）
        # Downsampling path
        self.down1 = ConvBlock(1, down1_out_ch, time_embed_dim, film_cond_dim, f0_score_scale, use_note_mask, note_mask_scale)
        self.down2 = ConvBlock(
            down1_out_ch, down2_out_ch, time_embed_dim, film_cond_dim, f0_score_scale, use_note_mask, note_mask_scale
        )
        self.down3 = ConvBlock(
            down2_out_ch, down3_out_ch, time_embed_dim, film_cond_dim, f0_score_scale, use_note_mask, note_mask_scale
        )
        self.bot1 = ConvBlock(down3_out_ch, bot1_out_ch, time_embed_dim, film_cond_dim, f0_score_scale, use_note_mask, note_mask_scale)

        # Upsampling path
        self.up3 = ConvBlock(
            bot1_out_ch + down3_out_ch, down3_out_ch, time_embed_dim, film_cond_dim, f0_score_scale, use_note_mask, note_mask_scale
        )
        self.up2 = ConvBlock(
            down3_out_ch + down2_out_ch, down2_out_ch, time_embed_dim, film_cond_dim, f0_score_scale, use_note_mask, note_mask_scale
        )
        self.up1 = ConvBlock(
            down2_out_ch + down1_out_ch, down1_out_ch, time_embed_dim, film_cond_dim, f0_score_scale, use_note_mask, note_mask_scale
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
        note_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            x: (B, L, 18) - ノイズが加えられた合成パラメータ
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

        # (B, L, 18) -> (B, 1, 18, L) - 2次元画像として扱う
        # 高さ=18（特徴次元）、幅=L（時間軸）、チャネル=1
        B, L, D = x.shape
        original_L = L
        
        # MaxPool2dとUpsampleの整合性を保つため、長さを8の倍数にパディング
        # (3回プーリングするため、8=2^3)
        if L % 8 != 0:
            pad_length = 8 - (L % 8)
            # 最後の値を複製してパディング
            x = torch.cat([x, x[:, -1:, :].expand(-1, pad_length, -1)], dim=1)
            L = x.shape[1]
            # f0_scoreも同様にパディング
            f0_score = torch.cat([f0_score, f0_score[:, -1:].expand(-1, pad_length)], dim=1)
            # note_maskも同様にパディング
            if note_mask is not None:
                note_mask = torch.cat([note_mask, note_mask[:, -1:].expand(-1, pad_length)], dim=1)
        
        x = x.transpose(1, 2)  # (B, 18, L)
        x = x.unsqueeze(1)  # (B, 1, 18, L)

        # f0_scoreとnote_maskは時間変動を保持したまま各ConvBlockに渡す
        # Downsampling（高さは18で固定、時間軸のみダウンサンプリング）
        x1 = self.down1(x, t, film_cond, f0_score, note_mask)  # (B, down1_out_ch, 18, L)
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
        # note_maskもダウンサンプリング（時間軸方向のみ）
        note_mask_down1 = None
        if note_mask is not None:
            note_mask_down1 = (
                F.interpolate(
                    note_mask.unsqueeze(1).unsqueeze(1),
                    size=(1, x.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(1)
                .squeeze(1)
            )  # (B, L/2)

        x2 = self.down2(x, t, film_cond, f0_score_down1, note_mask_down1)  # (B, down2_out_ch, 18, L/2)
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
        # note_maskもダウンサンプリング
        note_mask_down2 = None
        if note_mask_down1 is not None:
            note_mask_down2 = (
                F.interpolate(
                    note_mask_down1.unsqueeze(1).unsqueeze(1),
                    size=(1, x.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(1)
                .squeeze(1)
            )  # (B, L/4)

        x3 = self.down3(x, t, film_cond, f0_score_down2, note_mask_down2)  # (B, down3_out_ch, 18, L/4)
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
        # note_maskもダウンサンプリング
        note_mask_down3 = None
        if note_mask_down2 is not None:
            note_mask_down3 = (
                F.interpolate(
                    note_mask_down2.unsqueeze(1).unsqueeze(1),
                    size=(1, x.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(1)
                .squeeze(1)
            )  # (B, L/8)

        # Bottleneck
        x = self.bot1(x, t, film_cond, f0_score_down3, note_mask_down3)  # (B, bot1_out_ch, 18, L/8)

        # Upsampling（高さは18で固定、時間軸のみアップサンプリング）
        x = self.upsample(x)  # (B, bot1_out_ch, 18, L/4)
        x = torch.cat([x, x3], dim=1)  # (B, bot1_out_ch + down3_out_ch, 18, L/4)
        x = self.up3(x, t, film_cond, f0_score_down2, note_mask_down2)  # (B, down3_out_ch, 18, L/4)

        x = self.upsample(x)  # (B, down3_out_ch, 18, L/2)
        x = torch.cat([x, x2], dim=1)  # (B, down3_out_ch + down2_out_ch, 18, L/2)
        x = self.up2(x, t, film_cond, f0_score_down1, note_mask_down1)  # (B, down2_out_ch, 18, L/2)

        x = self.upsample(x)  # (B, down2_out_ch, 18, L)
        x = torch.cat([x, x1], dim=1)  # (B, down2_out_ch + down1_out_ch, 18, L)
        x = self.up1(x, t, film_cond, f0_score, note_mask)  # (B, down1_out_ch, 18, L)

        # Output
        x = self.out(x)  # (B, 1, 18, L)

        # (B, 1, 18, L) -> (B, L, 18)
        x = x.squeeze(1)  # (B, 18, L)
        x = x.transpose(1, 2)  # (B, L, 18)
        
        # パディングした分を削除して元の長さに戻す
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
        use_diffusers: bool = False,
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        self.use_diffusers = use_diffusers
        
        if use_diffusers:
            # DDPMSchedulerを使用
            self.scheduler = DDPMScheduler(
                num_train_timesteps=num_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule="linear",
                prediction_type="epsilon",  # ノイズ予測（従来と同じ）
            )
            # 従来の計算も保持（互換性のため）
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
            self.alphas = 1 - self.betas
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        else:
            # 従来の実装
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

        if self.use_diffusers and self.scheduler is not None:
            # diffusersのDDPMSchedulerを使用
            # DDPMSchedulerのadd_noiseは (B, C, H, W) 形式を期待するため、形状を変換
            # x_0: (B, L, D) -> (B, 1, L, D) (画像として扱う: C=1, H=L, W=D)
            B, L, D = x_0.shape
            x_0_4d = x_0.unsqueeze(1)  # (B, 1, L, D)
            
            # ノイズを生成
            noise = torch.randn_like(x_0_4d, device=self.device)  # (B, 1, L, D)
            
            # DDPMSchedulerは0-indexedのtimestepsを期待するため、t-1に変換
            timesteps = t - 1  # 0-indexedに変換 (B,)
            
            # DDPMSchedulerのadd_noiseを使用（timestepsはバッチとして受け取れる）
            x_t_4d = self.scheduler.add_noise(
                original_samples=x_0_4d,
                noise=noise,
                timesteps=timesteps,  # (B,) - 各サンプルが異なるtimestepを持てる
            )  # (B, 1, L, D)
            
            # 形状を元に戻す
            x_t = x_t_4d.squeeze(1)  # (B, L, D)
            noise = noise.squeeze(1)  # (B, L, D)
            
            return x_t, noise
        
        # 従来の実装
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
        guidance_scale: float = 1.0,
        note_mask: torch.Tensor | None = None,
    ):
        """
        ノイズ除去（サンプリング時）

        Args:
            model: UNetDiffusionモデル
            x: (B, L, D) - ノイズが加えられた合成パラメータ
            t: (B,) - 時間ステップ (1-indexed)
            instrument_ids: (B,) - 楽器ID
            f0_score: (B, L) - f0スコア
            guidance_scale: CFGのガイダンススケール（1.0でCFG無効、>1.0で条件の影響を強化）
            note_mask: (B, L) or None - 音符存在マスク
        Returns:
            x_prev: (B, L, D) - 前のステップのサンプル
        """
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        if self.use_diffusers and self.scheduler is not None:
            # diffusersのDDPMSchedulerを使用
            with torch.no_grad():
                # モデルでノイズを予測
                if guidance_scale > 1.0:
                    # CFG: 条件付きと条件なしの両方を計算
                    N = x.size(0)
                    cond_mask = torch.ones(N, device=x.device, dtype=torch.bool)
                    eps_cond = model(x, t, instrument_ids, f0_score, instrument_cond_mask=cond_mask, note_mask=note_mask)
                    
                    uncond_mask = torch.zeros(N, device=x.device, dtype=torch.bool)
                    eps_uncond = model(x, t, instrument_ids, f0_score, instrument_cond_mask=uncond_mask, note_mask=note_mask)
                    
                    # CFG適用
                    model_output = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                else:
                    # CFG無効（通常の条件付き予測）
                    model_output = model(x, t, instrument_ids, f0_score, note_mask=note_mask)
            
            # DDPMSchedulerのstepメソッドを使用
            # x: (B, L, D) -> (B, 1, L, D) に変換
            B, L, D = x.shape
            x_4d = x.unsqueeze(1)  # (B, 1, L, D)
            model_output_4d = model_output.unsqueeze(1)  # (B, 1, L, D)
            
            # DDPMSchedulerは0-indexedのtimestepsを期待するため、t-1に変換
            timesteps = t - 1  # (B,)
            
            # バッチ内の全サンプルが同じtimestepを持つことを確認
            # 通常、サンプリング時は全て同じtimestepを使用するため
            assert (timesteps == timesteps[0]).all(), "All samples in batch must have the same timestep for DDPMScheduler.step"
            timestep = timesteps[0].item()  # intに変換
            
            # DDPMSchedulerのstepを使用（バッチ全体に対して動作）
            scheduler_output = self.scheduler.step(
                model_output=model_output_4d,
                timestep=timestep,  # int (バッチ内の最初のtimestep、全サンプルが同じ)
                sample=x_4d,
            )
            prev_sample = scheduler_output.prev_sample  # (B, 1, L, D)
            
            # 形状を元に戻す
            x_prev = prev_sample.squeeze(1)  # (B, L, D)
            
            return x_prev

        # 従来の実装
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
        # (N,) -> (N, 1, 1) にviewして、pred_x_0 (B, L, D) と掛け算できるようにする
        alpha = alpha.view(N, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1)
        beta = beta.view(N, 1, 1)

        with torch.no_grad():
            if guidance_scale > 1.0:
                # CFG: 条件付きと条件なしの両方を計算
                # 条件付き
                cond_mask = torch.ones(N, device=x.device, dtype=torch.bool)
                eps_cond = model(x, t, instrument_ids, f0_score, instrument_cond_mask=cond_mask, note_mask=note_mask)
                
                # 条件なし
                uncond_mask = torch.zeros(N, device=x.device, dtype=torch.bool)
                eps_uncond = model(x, t, instrument_ids, f0_score, instrument_cond_mask=uncond_mask, note_mask=note_mask)
                
                # CFG適用: eps_cond + guidance_scale * (eps_cond - eps_uncond)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                # CFG無効（通常の条件付き予測）
                eps = model(x, t, instrument_ids, f0_score, note_mask=note_mask)

        # 予測されたノイズから x_0 を推定
        pred_x_0 = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)

        # 前のステップへの遷移
        mu = (torch.sqrt(alpha_bar_prev) * beta / (1 - alpha_bar)) * pred_x_0 + (
            torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)
        ) * x

        std = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        std = std.view(N, 1, 1)

        noise = torch.randn_like(x, device=self.device)
        noise = noise * mask.view(N, 1, 1)  # t=1の場合はノイズを追加しない

        return mu + noise * std

    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        instrument_ids: torch.Tensor,
        f0_score: torch.Tensor,
        guidance_scale: float = 1.0,
        note_mask: torch.Tensor | None = None,
    ):
        """
        サンプリング

        Args:
            model: UNetDiffusionモデル
            shape: (B, L, D) - 生成したい形状
            instrument_ids: (B,) - 楽器ID
            f0_score: (B, L) - f0スコア
            guidance_scale: CFGのガイダンススケール（1.0でCFG無効、>1.0で条件の影響を強化）
            note_mask: (B, L) or None - 音符存在マスク
        """
        batch_size, _, _ = shape
        x = torch.randn(shape, device=self.device)

        pbar: tqdm = tqdm(desc="Sampling", total=self.num_timesteps)

        for i in range(self.num_timesteps, 0, -1):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t, instrument_ids, f0_score, guidance_scale=guidance_scale, note_mask=note_mask)
            pbar.set_postfix({"t": i})
            pbar.update(1)
        pbar.close()
        return x

def get_diffusion_model(
    model_type: str,
    in_dim: int = 18,
    time_embed_dim: int = 128,
    num_instruments: int = 10,
    film_cond_dim: int = 64,
    use_film: bool = True,
    use_cross_attention: bool = False,
    cfg_dropout_prob: float = 0.1,
    f0_score_scale: float = 1.0,
    use_note_mask: bool = False,
    note_mask_scale: float = 1.0,
    # UNet parameters
    down1_out_ch: int = 64,
    down2_out_ch: int = 128,
    down3_out_ch: int = 256,
    bot1_out_ch: int = 512,
    # DiT parameters
    dit_hidden_dim: int = 768,
    dit_num_layers: int = 12,
    dit_num_heads: int = 12,
    dit_mlp_ratio: float = 4.0,
    # diffusers関連
    use_diffusers: bool = False,
    sample_width: int | None = None,  # 横幅（時間軸の長さ、8の倍数にパディング後）
) -> nn.Module:
    """
    Factory function to create diffusion model based on config
    
    Args:
        model_type: "unet" or "dit"
        cfg_dropout_prob: CFG用のdropout確率（学習時に楽器ラベルを無効化する確率、0.1が推奨）
        use_diffusers: Trueの場合、diffusersのUNet2DModelを使用（model_type="unet"の場合のみ有効）
        その他のパラメータは各モデルの設定
    """
    if model_type.lower() == "unet":
        if use_diffusers:
            print(f"Using DiffusersUNet2DModelWrapper")
            return DiffusersUNet2DModelWrapper(
                in_dim=in_dim,
                time_embed_dim=time_embed_dim,
                num_instruments=num_instruments,
                film_cond_dim=film_cond_dim,
                cfg_dropout_prob=cfg_dropout_prob,
                f0_score_scale=f0_score_scale,
                use_note_mask=use_note_mask,
                note_mask_scale=note_mask_scale,
                sample_size=18,  # 特徴次元（高さ）
                sample_width=sample_width,  # 横幅（時間軸の長さ、8の倍数にパディング後）
                block_out_channels=(down1_out_ch, down2_out_ch, down3_out_ch),
            )
        else:
            return UNetDiffusion(
                in_dim=in_dim,
                time_embed_dim=time_embed_dim,
                num_instruments=num_instruments,
                film_cond_dim=film_cond_dim,
                down1_out_ch=down1_out_ch,
                down2_out_ch=down2_out_ch,
                down3_out_ch=down3_out_ch,
                bot1_out_ch=bot1_out_ch,
                cfg_dropout_prob=cfg_dropout_prob,
                f0_score_scale=f0_score_scale,
                use_note_mask=use_note_mask,
                note_mask_scale=note_mask_scale,
            )
    elif model_type.lower() == "dit":
        if use_diffusers:
            raise ValueError("use_diffusers=True is only supported for model_type='unet'")
        return DiTDiffusion(
            in_dim=in_dim,
            time_embed_dim=time_embed_dim,
            num_instruments=num_instruments,
            film_cond_dim=film_cond_dim,
            hidden_dim=dit_hidden_dim,
            num_layers=dit_num_layers,
            num_heads=dit_num_heads,
            mlp_ratio=dit_mlp_ratio,
            use_film=use_film,
            use_cross_attention=use_cross_attention,
            cfg_dropout_prob=cfg_dropout_prob,
            f0_score_scale=f0_score_scale,
            use_note_mask=use_note_mask,
            note_mask_scale=note_mask_scale,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'unet' or 'dit'")
