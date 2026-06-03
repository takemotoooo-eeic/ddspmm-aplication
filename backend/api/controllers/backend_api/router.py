from fastapi import APIRouter

from .ddsp import ddsp_router
from .diffusion import diffusion_router
from .fluidsynth import fluidsynth_router
from .melodyflow import melodyflow_router

backend_api_router = APIRouter()
backend_api_router.include_router(ddsp_router)
backend_api_router.include_router(diffusion_router)
backend_api_router.include_router(fluidsynth_router)
backend_api_router.include_router(melodyflow_router)
