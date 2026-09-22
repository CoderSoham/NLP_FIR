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


# ---- FEAT-014: the floor ----------------------------------------------------
# Measured on the fourteen labelled calls: the extraction model alone scores
# 10/14 with four under-triages; with this floor, 13/14 with none.

from utils.severity import (SIGNAL_BAND, apply_severity_floor,
                            severity_from_signals)


@pytest.mark.parametrize("transcript,expected", [
    ("he has been shot in the chest", "critical"),
    ("she is not breathing, I am doing cpr", "critical"),
    ("the house is on fire, I can see flames", "high"),
    ("my ex-boyfriend is threatening me", "high"),
    ("there has been a car crash on the highway", "medium"),
])
def test_a_transcript_implies_a_floor(transcript, expected):
    assert severity_from_signals(transcript) == expected


def test_a_transcript_implying_nothing_has_no_floor():
    assert severity_from_signals("my cat is stuck in a tree") is None
    assert severity_from_signals("") is None
    assert severity_from_signals(None) is None


def test_the_highest_implied_band_wins():
    """A call mentioning both a crash and a gun is a gun call."""
    assert severity_from_signals(
        "there was a crash and then someone pulled a gun") == "critical"


def test_the_floor_raises_an_under_called_severity():
    """The failure this exists to prevent: the pipeline said medium for an
    officer-involved shooting."""
    assert apply_severity_floor("medium", "officer down, shots fired") == "critical"
    assert apply_severity_floor("low", "she is not breathing") == "critical"


def test_the_floor_never_lowers_a_severity():
    """Over-triage sends too much; under-triage sends too little to someone
    who is dying. The combiner can only ever move upward."""
    assert apply_severity_floor("critical", "there was a minor crash") == "critical"
    assert apply_severity_floor("high", "my cat is in a tree") == "high"


def test_the_floor_leaves_a_call_with_no_signals_alone():
    assert apply_severity_floor("low", "my cat is in a tree") == "low"


@pytest.mark.parametrize("severity", [None, "", "nonsense"])
def test_an_unusable_severity_still_gets_the_floor(severity):
    assert apply_severity_floor(severity, "he has been shot") == "critical"


def test_an_unusable_severity_with_no_floor_is_unknown_not_invented():
    assert apply_severity_floor(None, "my cat is in a tree") == "unknown"
    assert apply_severity_floor("nonsense", "hello") == "unknown"


def test_every_signal_group_maps_to_a_band():
    """A group the pipeline matches but this does not map contributes no
    floor at all, silently."""
    from utils.signals import SIGNAL_GROUPS

    for name, _words, _actions in SIGNAL_GROUPS:
        assert name in SIGNAL_BAND, f"signal group {name!r} implies no severity"


def test_high_is_reachable_only_with_every_signal_at_maximum():
    """`high` needs 0.7 and the ceiling is 1.0, so the branch is live in
    theory. In practice it needs a confident high label, fully-confident NEG
    sentiment, a distress emotion and four relevant entities *at once* -- and
    across fourteen real calls the score never exceeded 0.64. That gap is why
    every one of the scorer's ten failures was an under-triage, and why it is
    now floored rather than trusted. See scripts/calibrate_severity.py."""
    from utils.severity import HIGH_THRESHOLD, severity_score

    best, _parts = severity_score(
        {"high": 1.0, "medium": 0.0, "low": 0.0},
        {"label": "NEG", "score": 1.0},
        {"label": "fear", "score": 1.0},
        [{"label": "PERSON"}] * 4)
    assert best > HIGH_THRESHOLD

    # A realistic call: the zero-shot classifier never exceeded 0.50 on any
    # of the fourteen, and neutral sentiment and emotion are the common case.
    realistic, _parts = severity_score(
        {"high": 0.50, "medium": 0.30, "low": 0.20},
        {"label": "NEU", "score": 0.9},
        {"label": "neutral", "score": 0.6},
        [{"label": "PERSON"}] * 4)
    assert realistic < HIGH_THRESHOLD


def test_the_floor_reports_the_words_that_raised_it():
    """Keyword matching cannot tell a threat from a figure of speech. On one
    labelled call a dispatcher's sarcastic "shoot her" floors a squabble
    between two teenagers at critical -- the single over-triage in the
    measured set. Showing the words lets a reader discount it in a second."""
    band, words = severity_from_signals(
        "did you want us to come over and shoot her?", with_evidence=True)
    assert band == "critical"
    assert "shoot" in words


def test_evidence_lists_only_the_words_behind_the_winning_band():
    band, words = severity_from_signals(
        "there was a car crash and then he was shot", with_evidence=True)
    assert band == "critical"
    assert "shot" in words and "car" not in words


def test_evidence_is_empty_when_nothing_fires():
    assert severity_from_signals("the cat is in a tree",
                                 with_evidence=True) == (None, [])
