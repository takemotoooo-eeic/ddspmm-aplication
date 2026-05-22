# OpenAPI スキーマ（自動生成）

`openapi.yml` は **手で編集しない** ファイルです。

FastAPI のルート定義（`ddsp.py`, `diffusion.py`, `fluidsynth.py` など）と
`models.py` の Pydantic モデルから、`create_openapi_schema.py` が生成します。

フロントの [orval](https://orval.dev/) はこの YAML を読み、`frontend/src/orval/` に TypeScript クライアントを生成します。

## 更新手順

### 1. バックエンドで OpenAPI を再生成

```bash
# リポジトリルートから（api コンテナ起動後）
docker compose exec api python -m api.commands.create_openapi_schema

# または backend ディレクトリで
cd backend
PYTHONPATH=. python -m api.commands.create_openapi_schema
```

### 2. フロントで orval を再生成

```bash
cd frontend
npm run orval
```

`compose.yaml` では `openapi.yml` を web コンテナの `/opt/backend_api_openapi.yml` にマウントしているため、
**手順 1 のあと** フロントコンテナを使う場合はファイルが共有されます。

## API を変えるとき

| やること | 編集する場所 |
|---------|-------------|
| エンドポイント追加・変更 | `api/controllers/backend_api/*.py` |
| リクエスト/レスポンス型 | `api/controllers/backend_api/openapi/models.py` |
| OpenAPI 反映 | 上記スクリプトを実行 |
| フロント型・フック | `npm run orval` |

`openapi.yml` を直接書き換えても、次回スクリプト実行時に上書きされます。
