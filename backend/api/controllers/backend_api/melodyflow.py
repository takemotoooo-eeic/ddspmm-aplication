from io import BytesIO

from fastapi import APIRouter, File, Form, UploadFile

from api.controllers.common import WAV_RESPONSE, WAVResponse
from api.libs.exceptions import BadRequest
from api.libs.wav import verify_wav_file_format
from api.models.melodyflow import MelodyFlowEditParams, get_melodyflow_model

melodyflow_router = APIRouter()


@melodyflow_router.post(
    "/melodyflow/edit",
    responses={200: WAV_RESPONSE},
    response_class=WAVResponse,
)
async def edit_audio_with_melodyflow(
    wav_file: UploadFile = File(..., description="編集対象のWAVファイル"),
    start_sec: float = Form(..., description="編集区間の開始秒"),
    end_sec: float = Form(..., description="編集区間の終了秒"),
    text: str = Form(..., description="テキストプロンプト"),
) -> WAVResponse:
    try:
        verify_wav_file_format(wav_file)
        if not text.strip():
            raise BadRequest("text は空にできません。")

        wav_bytes = await wav_file.read()
        model = get_melodyflow_model()
        params = MelodyFlowEditParams(
            start_sec=start_sec,
            end_sec=end_sec,
            text=text.strip(),
        )
        edited = model.edit_region(wav_bytes, params)
        return WAVResponse(content=BytesIO(edited))
    except BadRequest:
        raise
    except ValueError as e:
        raise BadRequest(str(e))
    except Exception as e:
        raise BadRequest(str(e))
