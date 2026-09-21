"""Disagreements between the two analyses.

Imports nothing. These tests previously lived in test_llm.py and pulled
`compare_classifications` out of utils.audio_utils, so they ran only on a
machine that already had torch -- which meant CI could not check them, and the
first time they ran anywhere was after the bug they cover had shipped.
"""
from utils.compare import compare_classifications

BOTH = {"emergency_type": "police", "severity": "high",
        "probable_location": "Bannister", "response": {}}
RECORD = {"incident_type": "police", "severity": "high",
          "location": "1331 Bannister Road", "weapons": []}


def test_agreement_reports_nothing():
    assert compare_classifications(BOTH, RECORD) == []


def test_no_record_reports_nothing():
    assert compare_classifications(BOTH, None) == []
    assert compare_classifications(BOTH, {}) == []


def test_a_differing_type_is_reported():
    found = compare_classifications(BOTH, dict(RECORD, incident_type="fire"))
    assert found == [{"field": "emergency_type",
                      "classical": "police", "llm": "fire"}]


def test_an_unknown_type_is_not_a_disagreement():
    """The model declining to classify is not a second opinion."""
    for said in ("unknown", "other"):
        assert compare_classifications(
            BOTH, dict(RECORD, incident_type=said)) == []


def test_a_differing_severity_is_reported():
    found = compare_classifications(BOTH, dict(RECORD, severity="low"))
    assert found == [{"field": "severity", "classical": "high", "llm": "low"}]


# machine with torch installed, so CI could not check them at all.

def test_no_disagreement_is_reported_when_both_sides_came_from_the_model():
    """Regression: the classical severity can now be the model's own, and the
    'critical' -> 'high' fold was applied to one side only -- so a call the
    model called critical was reported as disagreeing with itself."""

    classical = {"emergency_type": "police", "severity": "critical",
                 "probable_location": "Bannister", "response": {}}
    record = {"incident_type": "police", "severity": "critical",
              "location": "1331 Bannister Road", "weapons": []}
    assert compare_classifications(classical, record,
                                   second_opinion=False) == []


def test_the_severity_fold_is_applied_to_both_sides():

    classical = {"emergency_type": "police", "severity": "critical",
                 "probable_location": "x", "response": {}}
    record = {"incident_type": "police", "severity": "high",
              "location": "x", "weapons": []}
    # critical and high are the same band on the classical scale.
    assert compare_classifications(classical, record) == []


def test_weapons_and_location_are_still_compared_without_a_second_opinion():
    """Those come from the keyword matcher and spaCy, which run either way."""

    classical = {"emergency_type": "police", "severity": "high",
                 "probable_location": None,
                 "response": {"signals": {"weapons": ["gun"]}}}
    record = {"incident_type": "police", "severity": "high",
              "location": "1331 Bannister Road", "weapons": []}
    fields = {d["field"] for d in
              compare_classifications(classical, record, second_opinion=False)}
    assert fields == {"weapons", "location"}
