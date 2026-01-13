"""
Diffusionモデルを使用して合成パラメータを生成するAPI
"""
from api.controllers.backend_api.openapi import models
from api.libs.exceptions import BadRequest
from api.models.diffusion import DiffusionModel
from api.models.diffusion.diffusion_model import DiffusionGenerateParams
from api.models.midi_aligner.midi_aligner import Note as MidiAlignerNote

from fastapi import APIRouter

diffusion_router = APIRouter()


@diffusion_router.post("/diffusion/generate", response_model=models.DDSPGenerateParams)
def generate_params_from_diffusion(params: models.DiffusionGenerateParams):
    """
    音符列と楽器IDから合成パラメータを生成（Diffusionモデルを使用）
    """
    try:

        diffusion_model = DiffusionModel()
        # models.NoteをMidiAlignerNoteに変換
        midi_notes = [
            MidiAlignerNote(
                start=note.start,
                frequency=note.frequency,
                duration=note.duration,
            )
            for note in params.notes
        ]
        
        # DiffusionGenerateParamsに変換
        diffusion_params = DiffusionGenerateParams(
            notes=midi_notes,
            instrument_name=params.instrument_name,
            signal_length=params.signal_length,
        )
        
        result = diffusion_model.generate(diffusion_params)
        
        # DDSPGenerateParams形式で返す
        return models.DDSPGenerateParams(
            pitch=result["pitch"],
            loudness=result["loudness"],
            z_feature=result["z_feature"],
        )
    except Exception as e:
        raise BadRequest(str(e))


