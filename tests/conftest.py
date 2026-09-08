"""Test fixtures.

The point of this file is that `import app` must work **without the ML stack.**
`utils.audio_utils` pulls in torch, whisper, spacy, transformers and librosa --
about 4.5 GB of weights on first use. Stubbing that one module lets the routes,
the upload validation and the template be tested in milliseconds.

What is *not* stubbed: app.py, config.py and templates/index.html are the real
files. These tests fail if those break.
"""
import io
import os
import shutil
import sys
import types
import uuid

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


class FakePipeline:
    """Stands in for utils.audio_utils. Records what it was asked to produce."""

    def __init__(self):
        self.report_id = uuid.uuid4().hex
        self.plots = ["waveform", "mfcc", "pitch", "entities"]
        self.output_folder = None

    def process_audio_file(self, input_path, output_folder):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        for kind in self.plots:
            path = os.path.join(output_folder, f"{kind}_{self.report_id}.png")
            with open(path, "wb") as fh:
                fh.write(b"\x89PNG\r\n\x1a\n" + b"0" * 256)
        return {
            "report_id": self.report_id,
            "plots": list(self.plots),
            "transcription": "there is a fire on alameda",
            "translated_text": None,
            "language": "en",
            "timestamp": "2026-01-01 00:00:00",
            "emergency_type": "fire",
            "severity": "high",
            "emergency_response": {
                "priority": "high",
                "suggestions": ["Dispatch fire department immediately"],
            },
            "summary": "A fire was reported on Alameda.",
            "best_match": "there is a fire",
            "score": 0.91,
            "entities": [{"text": "Alameda", "label": "LOC"}],
        }


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Install the stub, point storage at tmp_path, and hand back the app."""
    fake = FakePipeline()
    module = types.ModuleType("utils.audio_utils")
    module.process_audio_file = fake.process_audio_file
    module.generate_fir_pdf = lambda data: "unused.pdf"

    uploads = tmp_path / "uploads"
    processed = tmp_path / "processed"
    uploads.mkdir()
    processed.mkdir()

    monkeypatch.setenv("UPLOAD_FOLDER", str(uploads))
    monkeypatch.setenv("PROCESSED_FOLDER", str(processed))
    monkeypatch.setitem(sys.modules, "utils.audio_utils", module)

    for name in ("app", "config", "utils"):
        sys.modules.pop(name, None)

    import utils
    utils.audio_utils = module
    import app as app_module

    app_module.app.config["UPLOAD_FOLDER"] = str(uploads)
    app_module.PROCESSED_FOLDER = str(processed)
    app_module.app.config["TESTING"] = True

    fake.client = app_module.app.test_client()
    fake.uploads = uploads
    fake.processed = processed
    yield fake

    for name in ("app", "config"):
        sys.modules.pop(name, None)


def upload(client, name="call.mp3", data=b"fake audio"):
    return client.post(
        "/",
        data={"audio_file": (io.BytesIO(data), name)},
        content_type="multipart/form-data",
    )
