from fastapi import APIRouter, File, UploadFile
from typing import Generator
import json
from api.controllers.backend_api.openapi import models
from api.controllers.common import WAV_RESPONSE, WAVResponse
from api.libs.exceptions import BadRequest
from api.libs.midi import verify_mid_file_format
from api.libs.wav import verify_wav_file_format
from api.models.ddsp import DDSPModel, TrainInput
from api.models.midi_aligner import MidiAligner
from io import BytesIO

from fastapi.responses import StreamingResponse

ddsp_router = APIRouter()


@ddsp_router.post("/ddsp/train", response_model=models.Features)
async def train_ddsp(
    wav_file: UploadFile = File(..., description="WAVファイル"),
    midi_file: UploadFile = File(..., description="MIDIファイル"),
) -> StreamingResponse:
    try:
        verify_wav_file_format(wav_file)
        verify_mid_file_format(midi_file)

        midi_file_bytes = await midi_file.read()
        wav_file_bytes = await wav_file.read()

        midi_aligner = MidiAligner()
        aligned_midi_list, num_instruments, instrument_names = midi_aligner.align(
            wav_file_bytes, midi_file_bytes
        )

        ddsp_model = DDSPModel()
        train_input = TrainInput(
            wav_file=wav_file_bytes,
            num_instruments=num_instruments,
            instrument_names=instrument_names,
            midi=aligned_midi_list,
        )
        stream_output = ddsp_model.train(train_input)
        return StreamingResponse(_format_stream_response(stream_output), media_type="application/jsonline")
    except Exception as e:
        raise BadRequest(e)
    
def _format_stream_response(generator: models.TrainDDSPOutputStream) -> Generator[str, None, None]:
    for item in generator:
        if isinstance(item, models.TrainingProgress):
            yield json.dumps({"current_epoch": item.current_epoch, "total_epochs": item.total_epochs, "loss": item.loss}) + "\n"
        elif isinstance(item, models.Features):
            yield item.model_dump_json() + "\n"

@ddsp_router.post(
    "/ddsp/generate",
    responses={200: WAV_RESPONSE},
    response_class=WAVResponse,
)
def generate_audio_from_ddsp(params: models.DDSPGenerateParams):
    ddsp_model = DDSPModel()
    wav_data: bytes = ddsp_model.generate(
        pitch=params.pitch,
        loudness=params.loudness,
        z_feature=params.z_feature,
    )
    return WAVResponse(content=BytesIO(wav_data))
