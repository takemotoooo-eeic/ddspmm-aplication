from io import BytesIO

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse

from api.controllers.backend_api.openapi import models
from api.controllers.common import WAV_RESPONSE, WAVResponse, ZIP_RESPONSE, ZIPResponse
from api.libs.exceptions import BadRequest
from api.libs.instrument import parse_instrument_names_from_urmp_filename
from api.libs.midi import verify_mid_file_format
from api.libs.note import Note as MidiNote
from api.libs.wav import verify_wav_file_format
from api.models import MidiAligner
from api.models.fluidsynth import (
    FluidSynthGenerateParams,
    FluidSynthModel,
    FluidSynthTrainInput,
)

fluidsynth_router = APIRouter()


@fluidsynth_router.post(
    "/fluidsynth/train",
    responses={200: ZIP_RESPONSE},
    response_class=ZIPResponse,
)
async def train_fluidsynth(
    wav_file: UploadFile = File(..., description="WAVファイル"),
    midi_file: UploadFile = File(..., description="MIDIファイル"),
) -> StreamingResponse:
    """WAVとMIDIをアラインし、各楽器のアライン済みMIDIから FluidSynth で音源を生成してZIPで返す。"""
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
            if len(urmp_instrument_names) != len(aligned_midi_list):
                raise BadRequest(
                    "WAVファイル名から取得した楽器数とアライン済みトラック数が一致しません。"
                    f" (WAV: {urmp_instrument_names}, tracks: {len(aligned_midi_list)}件)"
                )
            instrument_names = urmp_instrument_names

        model = FluidSynthModel()
        train_input = FluidSynthTrainInput(
            aligned_midi_list=aligned_midi_list,
            instrument_names=instrument_names,
        )
        zip_bytes = model.train_to_zip(train_input)
        return ZIPResponse(content=BytesIO(zip_bytes))
    except Exception as e:
        raise BadRequest(str(e))


@fluidsynth_router.post(
    "/fluidsynth/generate",
    responses={200: WAV_RESPONSE},
    response_class=WAVResponse,
)
def generate_audio_from_fluidsynth(params: models.FluidsynthGenerateParams) -> WAVResponse:
    """音符列から MIDI を作成し、FluidSynth で音源を合成する。"""
    try:
        model = FluidSynthModel()
        fluidsynth_params = FluidSynthGenerateParams(
            notes=[
                MidiNote(
                    start=note.start,
                    frequency=note.frequency,
                    duration=note.duration,
                )
                for note in params.notes
            ],
            instrument_name=params.instrument_name,
            signal_length=params.signal_length,
        )
        wav_data = model.generate(fluidsynth_params)
        return WAVResponse(content=BytesIO(wav_data))
    except Exception as e:
        raise BadRequest(str(e))
