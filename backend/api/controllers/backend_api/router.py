from fastapi import APIRouter

from .ddsp import ddsp_router
from .diffusion import diffusion_router

backend_api_router = APIRouter()
backend_api_router.include_router(ddsp_router)
backend_api_router.include_router(diffusion_router)
