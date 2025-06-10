from uuid import UUID

from fastapi import Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.libs.logging import StructLogger, get_logger


class ErrorHandlers:
    logger: StructLogger

    def __init__(self):
        self.logger = get_logger()

    def _create_message(self, code: int, message: str):
        return {
            "code": code,
            "message": message,
        }

    # 400番台
    def handle_client_error(self, e: StarletteHTTPException):
        msg = self._create_message(e.status_code, e.detail)
        self.logger.warning(msg, exc_info=e)
        return JSONResponse(status_code=e.status_code, content={"error": e.detail or ""})

    # 500
    def handle_internal_error(self, e: Exception):
        msg = self._create_message(500, str(e))
        self.logger.error(msg, exc_info=e)
        return JSONResponse(status_code=500, content={"error": ""})

    def handle_error(self, reqeust: Request, e: Exception):
        if isinstance(e, ValidationError):
            return self.handle_validation_error(e)
        # 400番台
        if isinstance(e, StarletteHTTPException) and e.status_code >= 400 and e.status_code < 500:
            return self.handle_client_error(e)
        # 500
        return self.handle_internal_error(e)

    # validation error
    def handle_validation_error(self, e: ValidationError):
        msg = self._create_message(400, str(e))
        self.logger.warning(msg, exc_info=e)
        return JSONResponse(
            status_code=400,
            content={
                "error": "validation error",
                "details": [{"loc": error["loc"], "msg": error["msg"], "type": error["type"]} for error in e.errors()],
            },
        )

    def handle_request_validation_error(self, request: Request, e: RequestValidationError):
        msg = self._create_message(422, str(e))
        self.logger.warning(msg, exc_info=e)
        return JSONResponse(
            status_code=422,
            content={"error": "validation error", "details": jsonable_encoder(e.errors())},
        )

def http_exception_handler(request: Request, exc: Exception) -> Response:
    return ErrorHandlers().handle_error(request, exc)


def request_validation_exception_handler(request: Request, exc: Exception) -> Response:
    if isinstance(exc, RequestValidationError):
        return ErrorHandlers().handle_request_validation_error(request, exc)
    return ErrorHandlers().handle_error(request, exc)
