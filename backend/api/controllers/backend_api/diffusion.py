"""
Diffusionモデルを使用して合成パラメータを生成するAPI
"""
from fastapi import APIRouter, File, UploadFile

from api.controllers.backend_api.openapi import models
from api.libs.exceptions import BadRequest
from api.libs.instrument import parse_instrument_names_from_urmp_filename
from api.libs.midi import verify_mid_file_format
from api.libs.note import Note as MidiAlignerNote
from api.libs.wav import verify_wav_file_format
from api.models import DiffusionModel, DiffusionGenerateParams, DiffusionTrainInput, MidiAligner

diffusion_router = APIRouter()


@diffusion_router.post("/diffusion/train", response_model=models.Features)
async def train_diffusion(
    wav_file: UploadFile = File(..., description="WAVファイル"),
    midi_file: UploadFile = File(..., description="MIDIファイル"),
) -> models.Features:
    """
    WAVとMIDIをアラインし、DDSPガイダンス付きDiffusionで各楽器の合成パラメータを生成する。
    """
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

        diffusion_model = DiffusionModel()
        train_input = DiffusionTrainInput(
            wav_file=wav_file_bytes,
            num_instruments=num_instruments,
            instrument_names=instrument_names,
            midi=aligned_midi_list,
        )
        return diffusion_model.train(train_input)
    except Exception as e:
        raise BadRequest(str(e))


@diffusion_router.post("/diffusion/generate", response_model=models.DDSPGenerateParams)
def generate_params_from_diffusion(params: models.DiffusionGenerateParams):
    """
    音符列と楽器IDから合成パラメータを生成（Diffusionモデルを使用）
    """
    try:
        diffusion_model = DiffusionModel()
        midi_notes = [
            MidiAlignerNote(
                start=note.start,
                frequency=note.frequency,
                duration=note.duration,
            )
            for note in params.notes
        ]

        diffusion_params = DiffusionGenerateParams(
            notes=midi_notes,
            instrument_name=params.instrument_name,
            signal_length=params.signal_length,
            num_denoising_steps=params.num_denoising_steps,
            use_ddim=params.use_ddim,
            note_operation=params.note_operation,
            operation_prev_note=MidiAlignerNote(
                start=params.operation_prev_note.start,
                frequency=params.operation_prev_note.frequency,
                duration=params.operation_prev_note.duration,
            ) if params.operation_prev_note is not None else None,
            operation_note=MidiAlignerNote(
                start=params.operation_note.start,
                frequency=params.operation_note.frequency,
                duration=params.operation_note.duration,
            ) if params.operation_note is not None else None,
            prev_features=params.prev_features,
        )
        
        result = diffusion_model.generate(diffusion_params)

        return models.DDSPGenerateParams(
            pitch=result["pitch"],
            loudness=result["loudness"],
            z_feature=result["z_feature"],
        )
    except Exception as e:
        raise BadRequest(str(e))
