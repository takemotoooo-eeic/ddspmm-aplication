FROM node:20.11.0-slim AS base
WORKDIR /app

COPY ./frontend ./
RUN npm ci

# OpenAPI の生成
COPY ./backend/api/controllers/backend_api/openapi/openapi.yml /opt/backend_api_openapi.yml
RUN npm run orval

# 開発用
FROM base AS dev
RUN npm ci
EXPOSE 3000
CMD ["npm", "run", "dev"]
