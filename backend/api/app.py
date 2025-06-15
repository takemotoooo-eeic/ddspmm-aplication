from dotenv import load_dotenv
import os

env_path = ".env"
load_dotenv(env_path)

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware

from api.controllers.backend_api.router import backend_api_router
from api.libs.error_handlers import http_exception_handler, request_validation_exception_handler

app = FastAPI()

APP_URL = os.environ.get("APP_URL")


backend_api = FastAPI(
    title="backend_api",
    servers=[{"url": "http://localhost:8888/backend-api"}]
)
backend_api.add_exception_handler(StarletteHTTPException, http_exception_handler)
backend_api.add_exception_handler(RequestValidationError, request_validation_exception_handler)
backend_api.add_middleware(
    CORSMiddleware,
    allow_origins=[APP_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
backend_api.include_router(backend_api_router)


@app.get("/health", status_code=200)
def healthcheck():
    return {"status": "OK"}


app.mount("/backend-api", backend_api)
