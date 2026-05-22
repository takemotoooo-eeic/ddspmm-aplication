from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict


class CustomBaseModel(BaseModel):
    model_config = ConfigDict()


WAV_RESPONSE = {
    "content": {
        "audio/wav": {
            "schema": {
                "type": "string",
                "format": "binary",
            },
        },
    },
}


class WAVResponse(StreamingResponse):
    media_type = "audio/wav"


ZIP_RESPONSE = {
    "content": {
        "application/zip": {
            "schema": {
                "type": "string",
                "format": "binary",
            },
        },
    },
}


class ZIPResponse(StreamingResponse):
    media_type = "application/zip"
