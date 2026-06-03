# DDSPMM Application

混合音（WAV）とスコア（MIDI）から、楽器ごとの合成パラメータや音源を推定・編集・再合成する Web アプリケーションです。  
ブラウザ上のエディタで波形を確認しながら、ピッチ・ラウドネス・ノート（MIDI）を操作し、リアルタイムに近い形で聴き比べできます。

## アプリケーション概要

本システムは **フロントエンド（React）** と **バックエンド（FastAPI + PyTorch）** を Docker Compose で起動し、nginx がフロントと API をまとめて公開します。

| コンポーネント | 役割 |
|----------------|------|
| **Web UI** | トラック波形表示、3 モード切替、パラメータ編集、再生 |
| **API** | WAV/MIDI のアライン、学習（train）、合成（generate） |
| **nginx** | HTTPS 終端、フロントへのプロキシ、`/backend-api` への API プロキシ |

### 3 つの動作モード

左上のモード切替で、推定・編集・合成のパイプラインが変わります。

| モード | 概要 | Train | 再合成（Generate） | 主な編集 |
|--------|------|-------|-------------------|----------|
| **Diffusion+DDSPMM** | 拡散モデル＋DDSP ガイダンスでパラメータ推定。楽器変更やノート編集から拡散でパラメータを更新可能 | `POST /diffusion/train` | `diffusion/generate` → `ddsp/generate` | ノート（半音単位）、pitch/loudness 手描き、楽器選択 |
| **DDSPMM** | マルチスケールスペクトルロスで勾配降下しパラメータ推定 | `POST /ddsp/train` | `ddsp/generate` | pitch/loudness 手描きのみ |
| **FluidSynth** | 混合音は使わず、MIDI アライン結果を GM 音源でレンダリング | `POST /fluidsynth/train` | `fluidsynth/generate` | ピアノロール上のノートのみ |

- **Train（Import）**: WAV + MIDI をアップロード。学習用ハイパーパラメータ（lr / epoch 等）の UI はなく、サーバ側設定（`train.config.yaml` 等）に従います。
- **DDSPMM / Diffusion**: 推定した合成パラメータを **JSONL** で Export / Load できます（1 行 1 楽器）。
- **FluidSynth**: Train の応答は ZIP（各楽器の WAV + `manifest.json` にノート列）。

詳細な UI 仕様は [frontend/docs/FRONTEND_SPEC.md](frontend/docs/FRONTEND_SPEC.md) を参照してください。

### 典型的な利用フロー

1. モードを選ぶ（例: Diffusion+DDSPMM）
2. **Import** で混合 WAV と MIDI を指定 → 楽器トラックが波形付きで追加される
3. トラックを選択し、下部の編集パネルで **Edit** を有効化してパラメータやノートを変更
4. **REGENERATE** で再合成し、波形・再生で確認
5. （DDSP / Diffusion のみ）**Export** で JSONL を保存、または **Load** で以前のパラメータを読み込み

## システム構成

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────────┐
│   Browser   │────▶│ nginx :443   │────▶│ web (Vite/React) :3000  │
└─────────────┘     │              │     └─────────────────────────┘
                    │  /backend-api│────▶│ api (FastAPI) :8888     │
                    └──────────────┘     │  GPU推奨 (CUDA)          │
                                         └─────────────────────────┘
```

### バックエンド API 一覧

ベースパス: `/backend-api`（例: `http://localhost:8888/backend-api`）

| メソッド | パス | 説明 |
|----------|------|------|
| `POST` | `/ddsp/train` | WAV + MIDI → 各楽器の Feature（pitch, loudness, z_feature, notes） |
| `POST` | `/ddsp/generate` | Feature → WAV |
| `POST` | `/diffusion/train` | WAV + MIDI → DDSP ガイダンス付き Diffusion で Feature |
| `POST` | `/diffusion/generate` | ノート列 + 楽器名 → Feature |
| `POST` | `/fluidsynth/train` | WAV + MIDI → 楽器別 WAV の ZIP（+ manifest.json） |
| `POST` | `/fluidsynth/generate` | ノート列 + 楽器名 → WAV |

OpenAPI スキーマは FastAPI から自動生成します。手順は [backend/api/controllers/backend_api/openapi/README.md](backend/api/controllers/backend_api/openapi/README.md) を参照。

### 合成パラメータ（Feature）の形式

DDSP / Diffusion で扱う 1 楽器分の JSON オブジェクト（JSONL の 1 行）の例:

```json
{
  "instrument_name": "vn",
  "pitch": [0.0, 440.2, ...],
  "loudness": [-45.1, -38.0, ...],
  "z_feature": [[...], ...],
  "notes": [
    { "start": 0.5, "frequency": 440.0, "duration": 0.3 }
  ]
}
```

## 必要条件

- **Docker** / **Docker Compose**
- **NVIDIA GPU**（DDSP / Diffusion の train に推奨。`compose.yaml` の `api` サービスで GPU を予約）
- **FluidSynth モード**: ビルド前に SoundFont が必要（後述）

## セットアップ

### 1. SoundFont（FluidSynth 利用時）

API イメージのビルドには `FluidR3_GM.sf2` が必須です。

```bash
bash backend/scripts/download_soundfont.sh
```

配置先: `backend/api/models/fluidsynth_assets/soundfonts/FluidR3_GM.sf2`

**TTM (MelodyFlow)** 利用時は重みを事前に配置してください（約 4GB）。初回 API 呼び出し時の Hugging Face ダウンロードを避けられます。

配置先（サーバ上）: `backend/api/models/melodyflow_assets/melodyflow-t24-30secs/`  
必要なファイル: `compression_state_dict.bin`（約 230MB）, `state_dict.bin`（約 3.8GB）

**加えて** MelodyFlow はテキスト条件用 **T5 (`t5-base`)**（約 900MB）が必要です。ログが `T5 will be evaluated with autocast as float32` で止まって見える場合は、この取得待ちです。

T5 は **ホストの `backend/.cache/huggingface`** に落とし、`compose.yaml` が api コンテナの `/root/.cache/huggingface` にマウントします（コンテナ内ダウンロードは不要）。

```bash
cd ddspmm-aplication
bash backend/scripts/download_melodyflow_t5.sh
# 遅い / 認証が必要な場合
export HF_TOKEN=hf_xxxxxxxx
bash backend/scripts/download_melodyflow_t5.sh
docker compose up -d api
```

別パスに置く場合: `export HF_CACHE_HOST=/path/to/huggingface_cache` のあと `docker compose up -d api`。

#### T5 を手元で落としてサーバへ渡す

```bash
# 1. 手元（Mac など）で
cd ddspmm-aplication
bash backend/scripts/download_melodyflow_t5.sh

# 2. サーバへ転送
USER=your_user
HOST=your.server.example
REMOTE_DIR=/home/sarulab/kengo_takemoto/ddspmm-aplication/backend/.cache/huggingface
LOCAL_DIR=backend/.cache/huggingface

ssh "${USER}@${HOST}" "mkdir -p ${REMOTE_DIR}"
rsync -avP "${LOCAL_DIR}/" "${USER}@${HOST}:${REMOTE_DIR}/"

# 3. サーバで確認
ssh "${USER}@${HOST}" "bash /path/to/ddspmm-aplication/backend/scripts/download_melodyflow_t5.sh"
# → Already exists と出れば OK
```

#### 手早い回線のマシンで落として scp で渡す（推奨）

サーバの回線が遅いときは、自宅 PC などで取得してから転送するのが早いです。

**1. 手元（Mac / Linux）でダウンロード**

重みは **Model リポジトリ** [`facebook/melodyflow-t24-30secs`](https://huggingface.co/facebook/melodyflow-t24-30secs/tree/main) にあります。  
[MelodyFlow Space](https://huggingface.co/spaces/facebook/MelodyFlow)（`audiocraft/__init__.py` など）は **推論用コード** で、`state_dict.bin` は含まれません（`pyproject.toml` の git 依存と同じ中身）。

**おすすめ: 直リンクで `.bin` をそのまま保存**（`.cache` を作らない）

```bash
mkdir -p melodyflow-t24-30secs && cd melodyflow-t24-30secs
BASE=https://huggingface.co/facebook/melodyflow-t24-30secs/resolve/main

curl -fL -C - -o compression_state_dict.bin "${BASE}/compression_state_dict.bin"
curl -fL -C - -o state_dict.bin "${BASE}/state_dict.bin"
# 遅い場合: curl に -H "Authorization: Bearer $HF_TOKEN" を付ける
ls -lh *.bin
```

リポジトリのスクリプトでも同じ URL を使います（`aria2c` があれば並列）:

```bash
cd ddspmm-aplication
bash backend/scripts/download_melodyflow_model.sh
```

`huggingface-cli download --local-dir .` も使えますが、進捗は一旦 `.cache/huggingface/download/*.incomplete` に出ます。**完了後**にカレントへ `compression_state_dict.bin` / `state_dict.bin` が現れます。0% のまま止まる場合は回線か認証の問題です。

**2. サーバへ転送**（`USER` / `HOST` / `REMOTE_DIR` を環境に合わせて変更）

```bash
USER=your_user
HOST=your.server.example
REMOTE_DIR=/home/sarulab/kengo_takemoto/ddspmm-aplication/backend/api/models/melodyflow_assets/melodyflow-t24-30secs
LOCAL_DIR=backend/api/models/melodyflow_assets/melodyflow-t24-30secs

ssh "${USER}@${HOST}" "mkdir -p ${REMOTE_DIR}"

# rsync（途中で切れても -P で再開しやすい）
rsync -avP "${LOCAL_DIR}/"*.bin "${USER}@${HOST}:${REMOTE_DIR}/"

# scp でも可
# scp "${LOCAL_DIR}/"*.bin "${USER}@${HOST}:${REMOTE_DIR}/"
```

**3. サーバで確認**

```bash
ssh "${USER}@${HOST}" "ls -lh ${REMOTE_DIR}/*.bin"
# compression_state_dict.bin が ~200MB 以上、state_dict.bin が ~3.5GB 以上あれば OK

ssh "${USER}@${HOST}" "cd /path/to/ddspmm-aplication && bash backend/scripts/download_melodyflow_model.sh"
# → Already exists と出れば配置完了
```

#### サーバ上で直接ダウンロード

```bash
bash backend/scripts/download_melodyflow_model.sh
export HF_TOKEN=hf_xxxxxxxx   # 未認証で遅い場合
bash backend/scripts/download_melodyflow_model.sh
```

Docker では `compose.yaml` がこのディレクトリをマウントします。別パスに置く場合:

```bash
export MELODYFLOW_MODEL_HOST=/path/to/melodyflow-t24-30secs
docker compose up -d api
```

### 2. 環境変数

**フロントエンド**（`frontend/.env`）:

```env
VITE_API_BASE_URL="http://localhost:8888/backend-api"
VITE_APP_BASE_URL="http://localhost:3000"
```

本番で nginx 経由の場合は、実際のオリジンに合わせて変更してください。

**バックエンド**（`backend/.env`、任意）:

- `APP_URL`: CORS 許可オリジン（フロントの URL）

### 3. 起動

リポジトリルートで:

```bash
make init    # docker compose up --build -d
# または
docker compose up --build -d
```

| サービス | ポート（ホスト） | 説明 |
|----------|------------------|------|
| `web` | 3000 | 開発用 Vite サーバ |
| `api` | 8888 | FastAPI |
| `nginx` | 80, 443 | リバースプロキシ（本番向け） |

ヘルスチェック: `GET http://localhost:8888/health`

### 4. 停止・再起動

```bash
make stop    # docker compose stop
make reset   # down 後に init
```

## 使い方（Web UI）

1. ブラウザで `http://localhost:3000`（または nginx 経由の URL）を開く
2. 左上で **Diffusion+DDSPMM / DDSPMM / FluidSynth** を選択
3. 右端の **Import**（+）で WAV と MIDI を選びインポート
4. 中央の波形をクリックしてトラックを選択
5. 下部パネルで編集 → **REGENERATE** で再合成
6. ヘッダーの再生ボタンでミックス再生（ミュート・音量は左サイドバー）

### ヘッダーコントロール

| 操作 | 説明 |
|------|------|
| モード切替 | トラックはクリアされ、別パイプラインに切り替わる |
| ▶ / ■ | 全トラックの同期再生 / 停止 |
| Refresh | 全トラック削除 |
| Load（↑） | JSONL からパラメータ読込（DDSP / Diffusion のみ、トラック空のとき） |
| Export（↓） | 全トラックの Feature を JSONL ダウンロード（DDSP / Diffusion のみ） |
| Import（+） | WAV + MIDI で Train |

### エディター

- **Pitch**: ピッチ曲線は **連続** に手描き可能。Diffusion+DDSPMM ではノートは **半音単位** で移動
- **Loudness**: 表示範囲 -80 ～ -20 dB 固定、横スクロールのみ
- **FluidSynth**: ノートのピアノロールのみ（pitch / loudness 表示なし）

## リポジトリ構成

```
.
├── compose.yaml          # Docker Compose 定義
├── Makefile              # init / lint / oapigen など
├── nginx/                # リバースプロキシ設定
├── frontend/             # React + Vite + MUI
│   ├── src/
│   │   ├── api/          # バックエンド呼び出し
│   │   ├── components/   # UI コンポーネント
│   │   └── services/     # Import などの業務ロジック
│   └── docs/FRONTEND_SPEC.md
└── backend/
    ├── api/
    │   ├── app.py                    # FastAPI エントリ
    │   ├── controllers/backend_api/  # ddsp / diffusion / fluidsynth
    │   ├── models/                   # 学習・推論モデル
    │   ├── config/                   # train / loss 等の YAML
    │   └── libs/                     # MIDI アライン、FluidSynth など
    └── scripts/download_soundfont.sh
```

## 開発

### リンター・フォーマット

```bash
make lint
make format   # backend の ruff
```

### OpenAPI / フロント API クライアント再生成

API や `openapi/models.py` を変更したあと:

```bash
make oapigen
# backend: openapi.yml 生成
# frontend: src/orval/ 再生成
```

個別実行:

```bash
docker compose exec api python -m api.commands.create_openapi_schema
cd frontend && npm run orval
```

### フロントエンドのみ（ローカル）

```bash
cd frontend
npm ci
npm run dev
```

`openapi.yml` が無い場合は、先にバックエンドでスキーマ生成するか、compose の web サービス同様に YAML をマウントしてください。

### バックエンドの学習設定

Train 時の epoch 数・学習率などは UI からは変更しません。  
`backend/api/config/train.config.yaml` および関連する model 設定を編集してください。

## 関連ドキュメント

- [frontend/docs/FRONTEND_SPEC.md](frontend/docs/FRONTEND_SPEC.md) — フロントエンド UI・モード・エディター仕様
- [backend/api/controllers/backend_api/openapi/README.md](backend/api/controllers/backend_api/openapi/README.md) — OpenAPI 生成と orval 連携手順

## ライセンス・謝辞

学術・研究用途のプロジェクトを想定しています。  
DDSP、Diffusion、FluidSynth、URMP 形式のファイル名による楽器推定など、各サブシステムのライセンス・引用は利用するモデル・データセットの規約に従ってください。
