from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict


class CustomBaseModel(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(
            "model_family_",
            "model_families_",
        )
    )


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
