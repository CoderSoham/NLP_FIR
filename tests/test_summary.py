"""Summary length budgeting and the quality gate. No models."""
import pytest

from utils.summary import is_informative, length_budget

# From the real run of call_9.mp3 -- a domestic disturbance with no weapon,
# no vehicle and no injury. The old gate rejected every summary of it.
DOMESTIC = (
    "I need a police officer over here. I've got two teenage daughters and I "
    "just got home from work. They were physically fighting with each other "
    "and one of them kicked a hole in a door. They are 12 and almost 14 and "
    "the 12 year old is completely out of control and I cannot control this."
)


def test_budget_never_exceeds_the_input():
    """Regression: max_length=120 against an 82-token input made the model pad."""
    for tokens in (10, 40, 82, 149, 300):
        mx, mn = length_budget(tokens)
        assert mn < mx
        assert mx <= max(25, tokens) or tokens < 25


def test_budget_scales_with_input():
    assert length_budget(50)[0] < length_budget(300)[0]


def test_budget_is_capped():
    assert length_budget(100_000)[0] == 160


def test_budget_handles_empty_input():
    mx, mn = length_budget(0)
    assert mx > 0 and 0 < mn < mx


def test_a_faithful_summary_of_a_domestic_call_is_accepted():
    """Regression: the old gate needed a word from a weapons/medical whitelist,
    so a non-violent call could never produce an acceptable summary."""
    summary = ("Two teenage daughters were physically fighting at home and one "
               "kicked a hole in a door. The caller cannot control the 12 year old.")
    assert is_informative(summary, DOMESTIC)


def test_a_fragment_is_rejected():
    assert not is_informative("Police were called.", DOMESTIC)


def test_a_summary_longer_than_its_source_is_rejected():
    assert not is_informative(DOMESTIC + " " + DOMESTIC, DOMESTIC)


def test_fluent_but_unrelated_text_is_rejected():
    """The failure mode worth catching: an abstractive model that drifts."""
    invented = ("The quarterly financial results exceeded market expectations "
                "and shareholders approved the merger during the annual meeting.")
    assert not is_informative(invented, DOMESTIC)


def test_empty_inputs_are_rejected():
    assert not is_informative("", DOMESTIC)
    assert not is_informative(None, DOMESTIC)
    assert not is_informative("a real summary of something with many words here", "")


def test_gate_does_not_require_emergency_vocabulary():
    """It must not encode an assumption about the category of the call."""
    summary = ("The caller reports that a neighbour has been playing loud music "
               "late at night and refuses to stop when asked politely.")
    source = ("My neighbour has been playing loud music late at night for weeks "
              "and he refuses to stop when I ask him politely to turn it down.")
    assert is_informative(summary, source)
