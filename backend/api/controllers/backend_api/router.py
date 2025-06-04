from fastapi import APIRouter

from .ddsp import ddsp_router

backend_api_router = APIRouter()
backend_api_router.include_router(ddsp_router)
