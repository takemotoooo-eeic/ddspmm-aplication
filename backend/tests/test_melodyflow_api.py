import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from api.app import app
from api.libs.audio_splice import write_wav_bytes


def _make_wav_bytes(duration_sec: float = 2.0, sr: int = 22050) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return write_wav_bytes(audio, sr)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_melodyflow_edit():
    def _fake_edit(wav_bytes: bytes, params):
        full, sr = __import__(
            "api.libs.audio_splice", fromlist=["read_wav_bytes"]
        ).read_wav_bytes(wav_bytes)
        start = int(params.start_sec * sr)
        end = int(params.end_sec * sr)
        edited = full.copy()
        edited[start:end] = 0.0
        return write_wav_bytes(edited, sr)

    with patch("api.controllers.backend_api.melodyflow.get_melodyflow_model") as mock_get:
        model = MagicMock()
        model.edit_region.side_effect = _fake_edit
        mock_get.return_value = model
        yield model


def test_melodyflow_edit_returns_wav(client, mock_melodyflow_edit):
    wav_bytes = _make_wav_bytes()
    response = client.post(
        "/backend-api/melodyflow/edit",
        data={"start_sec": "0.5", "end_sec": "1.0", "text": "jazz piano"},
        files={"wav_file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    mock_melodyflow_edit.edit_region.assert_called_once()
    out, sr = sf.read(io.BytesIO(response.content))
    assert out.shape[0] > 0
    assert sr == 22050


def test_melodyflow_edit_rejects_empty_text(client, mock_melodyflow_edit):
    wav_bytes = _make_wav_bytes()
    response = client.post(
        "/backend-api/melodyflow/edit",
        data={"start_sec": "0.5", "end_sec": "1.0", "text": "   "},
        files={"wav_file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
    )
    assert response.status_code == 400


def test_melodyflow_edit_rejects_non_wav(client):
    response = client.post(
        "/backend-api/melodyflow/edit",
        data={"start_sec": "0.5", "end_sec": "1.0", "text": "test"},
        files={"wav_file": ("test.mp3", io.BytesIO(b"fake"), "audio/mpeg")},
    )
    assert response.status_code == 400
