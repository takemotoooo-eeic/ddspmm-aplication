# フロントエンド改修仕様書

## 1. 概要

DDSP エディタを 3 つの動作モードで切り替え可能にし、コンポーネントの再利用と責務分離を行う。学習（train）時のハイパーパラメータ UI と進捗表示は廃止する。

## 2. 動作モード

左上のボタン（`ModeSelector`）で以下 3 モードを切り替える。

| モード ID | 表示名 | Train API | Generate API | 編集内容 |
|-----------|--------|-----------|--------------|----------|
| `diffusion_ddsp` | Diffusion+DDSPMM | `POST /diffusion/train` | `POST /diffusion/generate` → `POST /ddsp/generate` | ノート（離散）・楽器変更・pitch/loudness 手描き |
| `ddsp` | DDSPMM | `POST /ddsp/train` | `POST /ddsp/generate` | pitch/loudness 手描きのみ |
| `fluidsynth` | FluidSynth | `POST /fluidsynth/train` | `POST /fluidsynth/generate` | ノート（離散）のみ。混合音は使用しない |

### 2.1 Diffusion+DDSPMM

- **Import**: WAV + MIDI を送信し `diffusion/train` で各楽器の合成パラメータ（Feature）を取得。続けて `ddsp/generate` で波形を生成してトラック追加。
- **編集パネル**: Pitch / Loudness タブ、楽器セレクト、REGENERATE（`diffusion/generate` → `ddsp/generate`）。
- **ノート**: ピアノロール上で離散的（半音単位）に移動。

### 2.2 DDSPMM

- **Import**: `ddsp/train` → `ddsp/generate`。
- **編集パネル**: Pitch / Loudness タブのみ。REGENERATE は `ddsp/generate` のみ。
- 楽器変更・diffusion generate は不可。

### 2.3 FluidSynth

- **Import**: `fluidsynth/train`（ZIP 返却）。混合音の波形は表示・再生しない。
- ZIP 内: 各楽器の `{instrument}.wav` と `manifest.json`（アライン済みノート列。バックエンド `train_to_zip` が同梱）。
- **編集パネル**: ピアノロール（ノートのみ）。Pitch / Loudness 表示なし。
- **REGENERATE**: `fluidsynth/generate`。

### 2.4 パラメータ読み込み（Load）・出力（Export）

- `diffusion_ddsp` / `ddsp` のみ JSONL から Feature を読み込み可能（Load）。
- 同モードで全トラックの推定パラメータを **JSONL** でダウンロード可能（Export）。1 行 1 楽器、Load と同じ Feature 形式。
- `fluidsynth` では Load / Export を非表示。

## 3. UI レイアウト

```
┌─────────────────────────────────────────────────────────────┐
│ [ModeSelector]  タイトル    再生時間 [▶][■]  [Refresh][Load][Import] │
├──────────┬──────────────────────────────────────────────────┤
│ Sidebar  │ タイムライン + 波形（トラックごと）                  │
│ (固定)   │                                                  │
├──────────┴──────────────────────────────────────────────────┤
│ 編集パネル（トラック選択時、リサイズ可能）                      │
│  [Pitch|Loudness] または [Notes only]  … REGENERATE [×]      │
└─────────────────────────────────────────────────────────────┘
```

- モード切替は AppBar **左端**（タイトル左またはタイトルと一体化）。
- ズーム UI は廃止。時間軸スケールは固定 `TIME_SCALE = 200` px/秒。

## 4. エディター仕様

### 4.1 Loudness（`diffusion_ddsp` / `ddsp` のみ）

| 項目 | 仕様 |
|------|------|
| dB 範囲 | **-80 ～ -20 dB 固定** |
| 縦スクロール | **なし**（表示高さ固定） |
| 横スクロール | あり（タイムラインと同期） |
| 拡大 | なし |

### 4.2 Pitch（`diffusion_ddsp` / `ddsp` のみ）

| 項目 | 仕様 |
|------|------|
| 縦・横スクロール | あり |
| 拡大 | なし（ノート高さ・鍵盤高さ固定） |
| ピッチ線編集 | **連続**（自由な周波数で手描き） |
| ノート | 半音グリッドにスナップ（連続音高不可） |

### 4.3 ノートピアノロール（`diffusion_ddsp` のノート編集 / `fluidsynth`）

- 共通コンポーネント `NotesPianoRoll`。
- `fluidsynth` では pitch 曲線・loudness を描画しない。
- ノート移動は `Math.round` による MIDI 番号スナップ。

## 5. Train（Import）ダイアログ

- WAV + MIDI ファイル選択のみ。
- **廃止**: epochs / lr 等の設定、学習進捗バー、epoch 表示。
- Import 中はボタン上のスピナーのみ（任意）。

## 6. ディレクトリ構成（目標）

```
src/
  api/           # モード別 train / generate
  constants/     # TIME_SCALE, dB 範囲, 鍵盤定義
  utils/         # 音高スナップ, 音声長計算
  types/         # AppMode, TrackData
  hooks/         # useAudioPlayback
  components/
    layout/      # ModeSelector, AppHeader
    editors/     # LoudnessEditor, PitchEditor, NotesPianoRoll, 共有部品
    editPanel/   # EditPanel（モードで分岐）
    dialogs/     # ImportTrackDialog, LoadTrackDialog
  pages/         # MainPage（App 本体）
```

## 7. API クライアント

Orval 生成コードに加え、`src/api/backend.ts` で以下を明示的に実装:

- `trainDiffusion`, `trainDdsp`, `trainFluidsynth`
- `generateDdspAudio`, `generateDiffusionParams`, `generateFluidsynthAudio`
- `parseFluidsynthZip`（manifest + wav 抽出）

`ddsp/train` は **wav_file / midi_file のみ**（epochs / lr なし）。

## 8. データモデル

```typescript
type AppMode = 'diffusion_ddsp' | 'ddsp' | 'fluidsynth';

interface TrackData {
  id: string;
  name: string;
  instrument: string;
  wavData: Blob;
  features?: Feature;      // ddsp / diffusion 用
  notes?: Note[];          // fluidsynth 用（features.notes でも可）
  signalLength?: number;   // generate 用サンプル数
  muted: boolean;
  volume: number;
}
```

## 9. 非機能要件

- 既存のダークテーマ・再生・ミュート・タイムラインクリック挙動は維持。
- コンポーネントは props でモード依存を注入し、重複 UI は共通化する。
