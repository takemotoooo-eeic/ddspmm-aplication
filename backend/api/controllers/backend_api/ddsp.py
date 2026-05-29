from io import BytesIO

from fastapi import APIRouter, File, UploadFile

from api.controllers.backend_api.openapi import models
from api.controllers.common import WAV_RESPONSE, WAVResponse
from api.libs.exceptions import BadRequest
from api.libs.instrument import parse_instrument_names_from_urmp_filename
from api.libs.midi import verify_mid_file_format
from api.libs.wav import verify_wav_file_format
from api.models import MidiAligner, TrainInput, get_ddsp_model

ddsp_router = APIRouter()


@ddsp_router.post("/ddsp/train", response_model=models.Features)
async def train_ddsp(
    wav_file: UploadFile = File(..., description="WAVファイル"),
    midi_file: UploadFile = File(..., description="MIDIファイル"),
) -> models.Features:
    try:
        verify_wav_file_format(wav_file)
        verify_mid_file_format(midi_file)

        midi_file_bytes = await midi_file.read()
        wav_file_bytes = await wav_file.read()

        midi_aligner = MidiAligner()
        aligned_midi_list, num_instruments, instrument_names = midi_aligner.align(
            wav_file_bytes, midi_file_bytes
        )

        urmp_instrument_names = parse_instrument_names_from_urmp_filename(wav_file.filename)
        if urmp_instrument_names is not None:
            if len(urmp_instrument_names) == len(instrument_names):
                instrument_names = urmp_instrument_names
            else:
                raise BadRequest(
                    "WAVファイル名から取得した楽器数とMIDIの楽器数が一致しません。"
                    f" (WAV: {urmp_instrument_names}, MIDI: {len(instrument_names)}件)"
                )

        ddsp_model = get_ddsp_model()
        train_input = TrainInput(
            wav_file=wav_file_bytes,
            num_instruments=num_instruments,
            instrument_names=instrument_names,
            midi=aligned_midi_list,
        )
        return ddsp_model.train(train_input)
    except Exception as e:
        raise BadRequest(str(e))

@ddsp_router.post(
    "/ddsp/generate",
    responses={200: WAV_RESPONSE},
    response_class=WAVResponse,
)
def generate_audio_from_ddsp(params: models.DDSPGenerateParams):
    ddsp_model = get_ddsp_model()
    wav_data: bytes = ddsp_model.generate(
        pitch=params.pitch,
        loudness=params.loudness,
        z_feature=params.z_feature,
    )
    return WAVResponse(content=BytesIO(wav_data))
