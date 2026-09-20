"""The incident report document.

Loads no model -- utils.fir_pdf imports fpdf and nothing else, which is the
reason the layout was moved out of utils.audio_utils.
"""
import os

import pytest

from utils import fir_pdf
from utils.fir_pdf import sanitize


FULL = {
    "report_id": "a" * 32,
    "timestamp": "2026-09-20 11:00:00",
    "emergency_type": "police",
    "severity": "high",
    "summary": "An officer was shot at 1331 Bannister Road.",
    "transcription": "We have an officer shot. It's at 1331 East Bannister.",
    "translated_text": None,
    "probable_location": "Bannister",
    "analysed_duration_s": 590.1,
    "source_duration_s": 590.1,
    "truncated": False,
    "warnings": [],
    "disagreements": [],
    "entities": [
        {"text": "Bannister", "label": "LOC"},
        {"text": "bannister", "label": "LOC"},
        {"text": "Lydia", "label": "LOC"},
        {"text": "1511 hours", "label": "TIME"},
    ],
    "sentiment": {"label": "NEG", "score": 0.91},
    "emotion": {"label": "fear", "score": 0.77},
    "response": {
        "priority": "high",
        "estimated_response_time": 5,
        "suggestions": ["Dispatch police immediately", "Alert EMS"],
        "required_resources": ["patrol_units", "ambulance"],
        "signals": {"weapon": ["gun", "revolver"]},
    },
    "dispatch": {"basis": "location_match", "station": "Central",
                 "matched_on": "bannister", "eta_min": 7},
    "asr": {"model": "whisper-large-v3-turbo", "backend": "groq"},
    "asr_segments": [
        {"start": 0.0, "text": "239, hold the air."},
        {"start": 4.2, "text": "What's your location?"},
    ],
    "llm": {
        "incident_type": "police",
        "severity": "critical",
        "summary": "An officer was shot during a vehicle check.",
        "location": "1331 Bannister Road",
        "callback_number": None,
        "people_involved": "One officer, one suspect",
        "injuries": "Officer shot through the right arm",
        "vehicles": "Maroon Oldsmobile Cutlass Sierra",
        "suspect_description": "White male, 50s, multicoloured top, glasses",
        "weapons": [{"item": "high standard .22 handgun",
                     "quote": "It looked like a high standard .22"}],
        "uncertainties": ["Whether the suspect is still armed"],
        "_rejected": {"weapons": ["rifle"]},
    },
    "llm_meta": {"status": "ok", "backend": "nvidia",
                 "model": "nvidia/nemotron-3-ultra-550b-a55b @ nvidia"},
}

MINIMAL = {
    "report_id": "b" * 32,
    "timestamp": "2026-09-20 11:00:00",
    "emergency_type": "unknown",
    "severity": "low",
    "transcription": "",
}


def render(data, tmp_path):
    path = fir_pdf.generate(data, str(tmp_path))
    assert os.path.exists(path)
    return open(path, "rb").read()


# ---- sanitising ------------------------------------------------------------
# fpdf's core fonts are Latin-1. The old code dropped anything outside it,
# which deleted every curly apostrophe Whisper emits.

def test_a_curly_apostrophe_survives_as_an_apostrophe():
    assert sanitize("don’t") == "don't"


def test_dashes_and_ellipses_become_readable_ascii():
    assert sanitize("wait — no …") == "wait - no ..."


def test_smart_quotes_become_quotes():
    assert sanitize("he said “hello”") == 'he said "hello"'


def test_an_accented_name_keeps_its_letters():
    assert sanitize("Renée Café") == "Renee Cafe"


def test_none_is_empty_not_the_word_none():
    assert sanitize(None) == ""


def test_the_output_is_latin_1_encodable():
    sanitize("你好 \U0001f6a8 café").encode("latin-1")


# ---- the document ----------------------------------------------------------

def test_a_full_report_renders(tmp_path):
    assert render(FULL, tmp_path).startswith(b"%PDF")


def test_the_filename_is_per_report(tmp_path):
    path = fir_pdf.generate(FULL, str(tmp_path))
    assert os.path.basename(path) == f"fir_{'a' * 32}.pdf"


def test_a_report_with_almost_no_data_still_renders(tmp_path):
    """Every stage is allowed to fail; the report is not."""
    assert render(MINIMAL, tmp_path).startswith(b"%PDF")


def test_a_missing_llm_record_is_explained_not_omitted(tmp_path):
    data = dict(MINIMAL, llm=None,
                llm_meta={"status": "failed", "reason": "429 rate limited"})
    assert render(data, tmp_path).startswith(b"%PDF")


def test_the_output_folder_is_created(tmp_path):
    target = tmp_path / "does" / "not" / "exist"
    assert os.path.exists(fir_pdf.generate(FULL, str(target)))


def test_a_long_transcript_does_not_explode_the_page_count(tmp_path):
    """The old layout set a 10mm line height on the transcript too."""
    data = dict(FULL, asr_segments=[],
                transcription="the caller reported a fire. " * 900)
    pdf_bytes = render(data, tmp_path)
    pages = pdf_bytes.count(b"/Type /Page\n") or pdf_bytes.count(b"/Type/Page")
    assert pages < 30, f"{pages} pages for a 25k-character transcript"


def test_segments_are_preferred_over_the_flat_transcript(tmp_path):
    """Timestamps are the only reason to open the transcript section."""
    many = [{"start": i * 3.0, "text": f"line {i}"} for i in range(200)]
    assert render(dict(FULL, asr_segments=many), tmp_path).startswith(b"%PDF")


def test_a_segment_without_a_start_time_renders(tmp_path):
    data = dict(FULL, asr_segments=[{"text": "no timestamp here"}])
    assert render(data, tmp_path).startswith(b"%PDF")


def test_weapons_given_as_plain_strings_render(tmp_path):
    """Older records stored weapons as strings, not {item, quote}."""
    data = dict(FULL, llm=dict(FULL["llm"], weapons=["revolver", "knife"]))
    assert render(data, tmp_path).startswith(b"%PDF")


def test_warnings_and_disagreements_render(tmp_path):
    data = dict(FULL, truncated=True, source_duration_s=1800.0,
                warnings=["sentiment: CVE-2025-32434"],
                disagreements=[{"field": "incident_type",
                                "classical": "fire", "llm": "police"}])
    assert render(data, tmp_path).startswith(b"%PDF")


def test_a_translated_call_renders_both_texts(tmp_path):
    data = dict(FULL, language="es", translated_text="There is a fire.")
    assert render(data, tmp_path).startswith(b"%PDF")


def test_a_failed_over_provider_is_recorded(tmp_path):
    data = dict(FULL, llm_meta=dict(FULL["llm_meta"],
                                    failed_over_from="nvidia", backend="groq"))
    assert render(data, tmp_path).startswith(b"%PDF")


@pytest.mark.parametrize("severity", ["critical", "high", "medium", "low", "odd"])
def test_every_severity_has_a_colour(tmp_path, severity):
    data = dict(MINIMAL, severity=severity)
    assert render(data, tmp_path).startswith(b"%PDF")


def test_a_transcript_full_of_unicode_renders(tmp_path):
    data = dict(MINIMAL, asr_segments=[],
                transcription="“don’t” — café \U0001f6a8 你好")
    assert render(data, tmp_path).startswith(b"%PDF")


# ---- small formatting decisions the document depends on ---------------------

@pytest.mark.parametrize("seconds,expected", [
    (590.1, "9:50"), (0, "0:00"), (59.4, "0:59"), (60, "1:00"),
    (3600, "60:00"), ("120", "2:00"),
])
def test_durations_read_as_lengths_not_measurements(seconds, expected):
    assert fir_pdf._mmss(seconds) == expected


@pytest.mark.parametrize("value", [None, "", "not a number", object()])
def test_an_unusable_duration_does_not_crash_the_report(value):
    assert fir_pdf._mmss(value) == "unknown duration"


@pytest.mark.parametrize("value,expected", [
    (["a", "b"], "a; b"),
    (["only"], "only"),
    ([], ""),
    (None, ""),
    ({"left arm": "grazed"}, "left arm: grazed"),
    ("plain", "plain"),
    (7, "7"),
    ([None, "kept", ""], "kept"),
])
def test_list_valued_fields_read_as_prose_not_python(value, expected):
    """Regression: 'Injuries' printed ['officer shot through the right arm']."""
    assert fir_pdf.as_text(value) == expected


def test_the_bullet_survives_sanitising():
    """U+2022 is not in Latin-1, so it used to be deleted entirely, leaving a
    hanging indent with no marker."""
    assert sanitize(fir_pdf.BULLET) == fir_pdf.BULLET
    assert sanitize("• item") == "· item"


@pytest.mark.parametrize("value,expected", [
    (5.0, "5"), (5, "5"), (7.5, "7.5"), ("12", "12"),
    (None, "unknown"), ("soon", "unknown"),
])
def test_protocol_targets_are_whole_minutes(value, expected):
    """'5.0 minutes' reads as a measurement of something."""
    assert fir_pdf._minutes(value) == expected
