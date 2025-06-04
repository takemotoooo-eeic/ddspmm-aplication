from fastapi import APIRouter, File, UploadFile

from api.controllers.backend_api.openapi import models
from api.controllers.common import WAV_RESPONSE, WAVResponse
from api.libs.exceptions import BadRequest
from api.libs.midi import verify_mid_file_format
from api.libs.wav import verify_wav_file_format
from api.models.ddsp import DDSPModel, TrainInput
from api.models.midi_aligner import MidiAligner

ddsp_router = APIRouter()


@ddsp_router.post("/ddsp/train", response_model=models.Features)
def ddsp_train_api(
    wav_file: UploadFile = File(..., description="WAVファイル"),
    midi_file: UploadFile = File(..., description="MIDIファイル"),
):
    try:
        verify_wav_file_format(wav_file)
        verify_mid_file_format(midi_file)

        midi_file_bytes = midi_file.read()
        wav_file_bytes = wav_file.read()

        midi_aligner = MidiAligner()
        aligned_midi_list, num_instruments = midi_aligner.align(
            wav_file_bytes, midi_file_bytes
        )

        ddsp_model = DDSPModel()
        train_input = TrainInput(
            wav_file=wav_file_bytes,
            num_instruments=num_instruments,
            midi=aligned_midi_list,
        )
        result = ddsp_model.train(train_input)
        features = models.Features(
            features=[
                models.Feature(
                    pitch=result.features[i].pitch,
                    loudness=result.features[i].loudness,
                    z_feature=result.features[i].z_feature,
                )
                for i in range(num_instruments)
            ]
        )
        return features
    except Exception as e:
        raise BadRequest(str(e))


@ddsp_router.post(
    "/ddsp/generate",
    responses={200: WAV_RESPONSE},
    response_class=WAVResponse,
)
def ddsp_generate_api(params: models.DDSPGenerateParams):
    wav_data: bytes = DDSPModel.generate(
        pitch=params.pitch,
        loudness=params.loudness,
        z_feature=params.z_feature,
        num_instruments=params.num_instruments,
    )
    return WAVResponse(content=wav_data)
