"""Which stages run, and when. FEAT-022's acceptance criteria live here.

Loads no model: utils/policy.py imports os and nothing else, which is the
point of separating the decision from the thing being decided about.
"""
import pytest

from utils import policy
from utils.policy import local_analysis_wanted, model_summary

GOOD = {"incident_type": "police", "severity": "critical",
        "summary": "An officer was shot at 1331 Bannister Road during a stop."}


@pytest.fixture(autouse=True)
def default_mode(monkeypatch):
    monkeypatch.delenv("LOCAL_ANALYSIS", raising=False)


# ---- the default: the model answered, so BART does not run -----------------

def test_a_usable_record_skips_the_local_classifiers():
    assert local_analysis_wanted(GOOD) is False


def test_no_record_runs_the_local_classifiers():
    """The floor. With no model there has to be a classification from
    somewhere, and this is where it comes from."""
    assert local_analysis_wanted(None) is True
    assert local_analysis_wanted({}) is True


@pytest.mark.parametrize("record", [
    {"severity": "high"},                       # no type
    {"incident_type": "fire"},                  # no severity
    {"incident_type": None, "severity": None},
    {"incident_type": "", "severity": ""},
])
def test_a_record_missing_either_field_runs_them(record):
    """A record that classified nothing is not a classification, whatever
    its status field said."""
    assert local_analysis_wanted(record) is True


# ---- the modes -------------------------------------------------------------

def test_always_runs_them_even_on_a_good_record(monkeypatch):
    monkeypatch.setenv("LOCAL_ANALYSIS", "always")
    assert local_analysis_wanted(GOOD) is True


def test_never_skips_them_even_with_no_record(monkeypatch):
    monkeypatch.setenv("LOCAL_ANALYSIS", "never")
    assert local_analysis_wanted(None) is False


def test_an_unrecognised_mode_falls_back_to_auto(monkeypatch):
    """A typo in an env var should not silently change what runs."""
    monkeypatch.setenv("LOCAL_ANALYSIS", "yes please")
    assert policy.mode() == "auto"
    assert local_analysis_wanted(GOOD) is False


@pytest.mark.parametrize("raw", ["ALWAYS", " always ", "Always"])
def test_modes_are_case_and_space_insensitive(monkeypatch, raw):
    monkeypatch.setenv("LOCAL_ANALYSIS", raw)
    assert policy.mode() == "always"


def test_an_explicit_choice_overrides_the_environment(monkeypatch):
    monkeypatch.setenv("LOCAL_ANALYSIS", "never")
    assert local_analysis_wanted(GOOD, chosen="always") is True


# ---- the summary gate ------------------------------------------------------

def test_a_real_summary_is_used():
    assert model_summary(GOOD).startswith("An officer was shot")


@pytest.mark.parametrize("record", [
    None, {}, {"summary": None}, {"summary": ""}, {"summary": "   "},
    {"summary": "Unknown."}, {"summary": "A fire."},
])
def test_a_missing_or_fragmentary_summary_falls_back_to_bart(record):
    assert model_summary(record) == ""


def test_the_gate_is_a_word_count_not_a_lexical_overlap_check():
    """A structured extraction cannot drift from its source the way an
    extractive summariser can, so rejecting a good abstractive summary for
    using different words would cost a minute of BART for a worse result."""
    paraphrase = {"summary": "A law enforcement officer sustained a gunshot "
                             "wound during a vehicle stop."}
    assert model_summary(paraphrase)


def test_whitespace_around_a_summary_does_not_count_as_content():
    assert model_summary({"summary": "  \n  one two three  \n "}) == ""
