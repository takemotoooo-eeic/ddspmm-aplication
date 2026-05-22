from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI
from fastapi.routing import APIRoute

from api.app import backend_api

OpenAPISchema = dict[str, Any]


def replace_const_with_enum(schema: OpenAPISchema):
    for item in schema.values():
        if isinstance(item, dict):
            if "const" in item:
                item["enum"] = [item["const"]]
                item["type"] = "string"
                del item["const"]
            replace_const_with_enum(item)


def snake_to_upper_camel(snake_str: str):
    words = snake_str.split("_")
    return "".join(word.capitalize() for word in words)


def create_operation_id(app: FastAPI):
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = snake_to_upper_camel(route.name)


def _openapi_output_path(api: FastAPI) -> Path:
    """Docker (/app) とローカル (backend/) の両方で書き込み可能なパスを返す。"""
    candidates = [
        Path(f"/app/api/controllers/{api.title}/openapi/openapi.yml"),
        Path(__file__).resolve().parents[1]
        / "controllers"
        / api.title
        / "openapi"
        / "openapi.yml",
    ]
    for path in candidates:
        if path.parent.exists() or path.parent.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
    return candidates[-1]


if __name__ == "__main__":
    apis = [backend_api]
    for api in apis:
        create_operation_id(api)
        openapi_schema = api.openapi()
        replace_const_with_enum(openapi_schema)

        out = _openapi_output_path(api)
        with out.open("w") as f:
            yaml.dump(openapi_schema, f, default_flow_style=False, allow_unicode=True)
        print(f"Wrote OpenAPI schema to {out}")
