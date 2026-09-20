"""Severity scoring. Pure arithmetic, no models."""
import pytest

from utils.severity import (SEVERITY_ENTITY_LABELS, relevant_entities,
                            severity_label, severity_score)

CONFIDENT_HIGH = {"high": 0.9, "medium": 0.07, "low": 0.03}
CONFIDENT_LOW = {"high": 0.03, "medium": 0.07, "low": 0.9}
NEUTRAL = {"label": "NEU", "score": 0.5}
CALM = {"label": "neutral", "score": 0.9}


def ents(*labels):
    return [{"text": f"t{i}", "label": l} for i, l in enumerate(labels)]


def test_confident_high_outscores_confident_low():
    """Regression: scores[0] was read without its label, so a confident 'low'
    scored identically to a confident 'high'."""
    hi, _ = severity_score(CONFIDENT_HIGH, NEUTRAL, CALM, [])
    lo, _ = severity_score(CONFIDENT_LOW, NEUTRAL, CALM, [])
    assert hi > lo


def test_bare_numbers_no_longer_inflate_severity():
    """Regression: 12 of 16 entities on a real call were CARDINAL/DATE --
    '911', ages, phone digits. Every call has those, so the term was a
    constant offset."""
    numbers = ents(*["CARDINAL"] * 8, *["DATE"] * 4)
    with_numbers, parts = severity_score(CONFIDENT_LOW, NEUTRAL, CALM, numbers)
    without, _ = severity_score(CONFIDENT_LOW, NEUTRAL, CALM, [])
    assert parts["entities"] == 0.0
    assert with_numbers == without


def test_relevant_entities_still_count():
    people_places = ents("PERSON", "LOC", "GPE")
    _, parts = severity_score(CONFIDENT_LOW, NEUTRAL, CALM, people_places)
    assert parts["entities"] > 0


def test_entity_term_is_capped():
    _, parts = severity_score(CONFIDENT_LOW, NEUTRAL, CALM, ents(*["PERSON"] * 50))
    assert parts["entities"] == 0.2


def test_low_confidence_emotion_is_ignored():
    """Regression: 'surprise' at 0.261 over 7 classes (chance 0.143) was
    weighted into the score with no threshold."""
    _, parts = severity_score(CONFIDENT_LOW, NEUTRAL,
                              {"label": "fear", "score": 0.26}, [])
    assert parts["emotion"] == 0.0


def test_confident_distress_emotion_counts():
    _, parts = severity_score(CONFIDENT_LOW, NEUTRAL,
                              {"label": "fear", "score": 0.88}, [])
    assert parts["emotion"] > 0


def test_non_distress_emotion_never_counts():
    _, parts = severity_score(CONFIDENT_LOW, NEUTRAL,
                              {"label": "joy", "score": 0.99}, [])
    assert parts["emotion"] == 0.0


def test_negative_sentiment_raises_the_score():
    with_neg, parts = severity_score(CONFIDENT_LOW,
                                     {"label": "NEG", "score": 0.9}, CALM, [])
    without, _ = severity_score(CONFIDENT_LOW, NEUTRAL, CALM, [])
    assert parts["sentiment"] > 0 and with_neg > without


@pytest.mark.parametrize("score,expected", [
    (0.0, "low"), (0.4, "low"), (0.41, "medium"),
    (0.7, "medium"), (0.71, "high"), (1.0, "high"),
])
def test_thresholds(score, expected):
    assert severity_label(score) == expected


def test_cardinal_and_date_are_excluded_by_design():
    assert "CARDINAL" not in SEVERITY_ENTITY_LABELS
    assert "DATE" not in SEVERITY_ENTITY_LABELS
    assert relevant_entities(ents("CARDINAL", "DATE")) == []


def test_missing_inputs_do_not_raise():
    assert severity_score({}, None, None, None)[0] >= 0.0
