"""Evidence verification: a claim must survive its own quote.

The driving case is call_9. Both a 1.5B and a 3B model reported a weapon on a
call with none -- `['gun']` from the dispatcher's sarcastic "shoot her", and
`['foot']` because a daughter kicked a door. A quote the model cannot produce
from the transcript is a hallucination, and that is checkable.
"""
import pytest

from utils.evidence import normalise, quote_supported, verify_evidence

TRANSCRIPT = (
    "911, what is your emergency? Yes, I need a police officer over here at 714 "
    "Court. They were physically fighting with each other, and one of them "
    "kicked a hole in a door. Okay, did you want us to come over to shoot her? "
    "Excuse me? That's a joke."
)


def test_a_real_quote_is_supported():
    assert quote_supported("kicked a hole in a door", TRANSCRIPT)


def test_punctuation_and_case_do_not_matter():
    """ASR punctuation is inconsistent; honest quotes must still match."""
    assert quote_supported("Kicked a hole in a door!!", TRANSCRIPT)
    assert quote_supported("i need a police officer", TRANSCRIPT)


def test_an_invented_quote_is_not_supported():
    assert not quote_supported("he is pointing a handgun at me", TRANSCRIPT)


def test_paraphrase_is_not_supported():
    """Tolerant of punctuation, intolerant of paraphrase."""
    assert not quote_supported("she kicked a door and made a hole", TRANSCRIPT)


@pytest.mark.parametrize("quote", ["gun", "shoot", "", None, "a"])
def test_single_words_are_not_evidence(quote):
    """'gun' appearing somewhere says nothing about a gun being present."""
    assert not quote_supported(quote, TRANSCRIPT)


def test_verify_keeps_supported_and_drops_invented():
    record = {"weapons": [
        {"item": "knife", "quote": "kicked a hole in a door"},
        {"item": "handgun", "quote": "he is pointing a handgun at me"},
    ]}
    kept, dropped = verify_evidence(record, TRANSCRIPT)
    assert [k["item"] for k in kept] == ["knife"]
    assert dropped[0]["item"] == "handgun"
    assert "not found" in dropped[0]["reason"]


def test_a_weapon_with_no_evidence_object_is_dropped():
    """The old schema returned bare strings; they carry no evidence at all."""
    kept, dropped = verify_evidence({"weapons": ["gun"]}, TRANSCRIPT)
    assert kept == []
    assert dropped[0]["reason"] == "no evidence supplied"


def test_empty_and_missing_fields_are_safe():
    assert verify_evidence({}, TRANSCRIPT) == ([], [])
    assert verify_evidence({"weapons": None}, TRANSCRIPT) == ([], [])


def test_normalise_collapses_whitespace_and_symbols():
    assert normalise("  He's   GOT a gun!! ") == "he s got a gun"
