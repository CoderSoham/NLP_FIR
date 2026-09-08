"""Transcript signal detection.

Every case here was taken from, or motivated by, a real run of the pipeline
over `static/audio/call_9.mp3` — a mother reporting her two daughters fighting.
"""
import pytest

from utils.signals import augment_actions, detect_signals

# The dispatcher's actual words on that call.
JOKE = "Okay, did you want us to come over to shoot her? Are you there? Excuse me?"


@pytest.mark.parametrize("text,word", [
    ("Excuse me, could you repeat that?", "excuse"),
    ("He went out the exit.", "exit"),
    ("What is the next street?", "next"),
    ("I'll send you a text.", "text"),
    ("For example, the blue one.", "example"),
    ("I'm expecting an ambulance.", "expecting"),
])
def test_ex_does_not_fire_on_words_containing_it(text, word):
    """'ex' was a keyword matched as a substring, so 'excuse me' -- which a
    dispatcher says on most calls -- raised domestic-violence actions."""
    assert "violence" not in detect_signals(text), f"{word!r} triggered violence"


@pytest.mark.parametrize("text", [
    "He is in cardiac arrest.",
    "She sounded really scared.",
    "I lost my card.",
])
def test_car_does_not_fire_on_words_containing_it(text):
    """'car' matched inside 'cardiac', so a cardiac arrest raised
    'Notify traffic control for scene safety'."""
    assert "vehicle" not in detect_signals(text), text


def test_cardiac_arrest_is_a_medical_signal_not_a_vehicle_one():
    signals = detect_signals("He is in cardiac arrest and not breathing.")
    assert "medical" in signals
    assert "vehicle" not in signals


@pytest.mark.parametrize("text,group", [
    ("There has been a car crash on the highway.", "vehicle"),
    ("I heard a gun shot.", "weapons"),
    ("She is not breathing.", "medical"),
    ("There is smoke coming from the kitchen.", "fire"),
    ("My ex-husband has a restraining order.", "violence"),
])
def test_real_signals_still_fire(text, group):
    assert group in detect_signals(text)


def test_signals_are_reported_with_their_evidence():
    """A reader must be able to see why an advisory was raised."""
    response = augment_actions(JOKE, {"suggestions": []})
    assert response["signals"]["weapons"] == ["shoot"]


def test_known_limit_keyword_matching_cannot_read_intent():
    """The sample call's firearm advisory came from a dispatcher's sarcasm.

    Whole-word matching does not fix this and is not meant to: 'shoot' really
    is in the transcript. The mitigation is that `signals` records the matched
    word, so the advisory is traceable to its evidence rather than presented as
    a judgement.
    """
    signals = detect_signals(JOKE)
    assert "weapons" in signals
    assert "violence" not in signals, "'Excuse me' must no longer imply violence"


def test_suggestions_are_deduplicated_in_order():
    base = {"suggestions": ["Dispatch police units", "Secure the area"]}
    out = augment_actions("He has a gun and a gun and a gun.", base)
    assert out["suggestions"][:2] == ["Dispatch police units", "Secure the area"]
    assert len(out["suggestions"]) == len(set(out["suggestions"]))


def test_empty_transcript_adds_nothing():
    assert detect_signals("") == {}
    assert detect_signals(None) == {}
