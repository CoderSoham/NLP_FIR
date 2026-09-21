"""Turning a location string into coordinates.

The default is the load-bearing part. A 911 transcript's location field is
the most identifying string the pipeline produces, so the tests that matter
most here are the ones asserting nothing leaves the machine.
"""
import json

import pytest

from utils import geocode
from utils.geocode import from_registry, locate, mode, normalise

REGISTRY = {
    "places": {
        "Bannister": {"lat": 38.9339, "lon": -94.5641},
        "East Bannister Road": {"lat": 38.9341, "lon": -94.5501},
        "Downtown": {"lat": 39.0997, "lon": -94.5786},
        "Broken": {"lat": "nonsense", "lon": None},
    }
}


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("GEOCODER", raising=False)
    geocode._CACHE.clear()


@pytest.fixture
def no_network(monkeypatch):
    """Any HTTP call in a test using this fixture is a failure, not a slow test."""
    def forbidden(*a, **k):
        raise AssertionError("the geocoder made a network call")
    monkeypatch.setattr(geocode.urllib.request, "urlopen", forbidden)


# ---- the default: off ------------------------------------------------------

def test_the_geocoder_is_off_by_default(no_network):
    assert mode() == "off"
    assert locate("1331 East Bannister Road", REGISTRY) is None


def test_an_unrecognised_mode_is_off(monkeypatch, no_network):
    """A typo in an env var must not start sending addresses to a third party."""
    monkeypatch.setenv("GEOCODER", "yes")
    assert mode() == "off"
    assert locate("1331 East Bannister Road", REGISTRY) is None


def test_registry_mode_never_reaches_the_network(monkeypatch, no_network):
    monkeypatch.setenv("GEOCODER", "registry")
    assert locate("1331 East Bannister Road", REGISTRY) == (38.9341, -94.5501)


def test_registry_mode_returns_nothing_rather_than_asking_anyone(monkeypatch, no_network):
    monkeypatch.setenv("GEOCODER", "registry")
    assert locate("somewhere nobody listed", REGISTRY) is None


def test_the_registry_is_tried_before_the_network_even_in_nominatim_mode(
        monkeypatch, no_network):
    """It costs nothing, sends nothing, and is more accurate for places an
    operator bothered to list."""
    monkeypatch.setenv("GEOCODER", "nominatim")
    assert locate("Downtown please", REGISTRY) == (39.0997, -94.5786)


# ---- the registry table ----------------------------------------------------

def test_the_longest_matching_place_wins():
    """'East Bannister Road' is more specific than 'Bannister', and someone
    took the trouble to add it."""
    assert from_registry("1331 East Bannister Road", REGISTRY) == (38.9341, -94.5501)


def test_a_shorter_place_still_matches_when_it_is_the_only_one():
    assert from_registry("out on Bannister", REGISTRY) == (38.9339, -94.5641)


def test_matching_ignores_case_and_punctuation():
    assert from_registry("DOWNTOWN, near the park!", REGISTRY) is not None


def test_a_place_with_unusable_coordinates_is_skipped():
    assert from_registry("Broken", REGISTRY) is None


@pytest.mark.parametrize("location", [None, "", "   "])
def test_an_empty_location_matches_nothing(location):
    assert from_registry(location, REGISTRY) is None


def test_a_registry_without_places_matches_nothing():
    assert from_registry("Bannister", {"fire": []}) is None
    assert from_registry("Bannister", None) is None


def test_normalise_survives_transcription_noise():
    assert normalise("1331 East Bannister Rd.") == normalise("1331 east bannister rd")


# ---- nominatim -------------------------------------------------------------

class _Resp:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return json.dumps(self.payload).encode()


@pytest.fixture
def instant(monkeypatch):
    """Nominatim's one-request-per-second rule is real; waiting for it is not."""
    monkeypatch.setattr(geocode.time, "sleep", lambda _s: None)
    monkeypatch.setattr(geocode.time, "monotonic", lambda: 10_000.0)


def test_nominatim_parses_a_result(monkeypatch, instant):
    monkeypatch.setenv("GEOCODER", "nominatim")
    monkeypatch.setattr(geocode.urllib.request, "urlopen",
                        lambda *a, **k: _Resp([{"lat": "38.93", "lon": "-94.56"}]))
    assert locate("nowhere in the table", REGISTRY) == (38.93, -94.56)


def test_nominatim_sends_an_identifying_user_agent(monkeypatch, instant):
    """Required by its usage policy; omitting it gets the project blocked."""
    seen = {}

    def capture(request, timeout=None):
        seen["ua"] = request.get_header("User-agent")
        return _Resp([])

    monkeypatch.setenv("GEOCODER", "nominatim")
    monkeypatch.setattr(geocode.urllib.request, "urlopen", capture)
    locate("nowhere in the table", REGISTRY)
    assert seen["ua"] and "nlp-fir" in seen["ua"]


def test_an_empty_result_is_none_not_an_error(monkeypatch, instant):
    monkeypatch.setenv("GEOCODER", "nominatim")
    monkeypatch.setattr(geocode.urllib.request, "urlopen",
                        lambda *a, **k: _Resp([]))
    assert locate("nowhere in the table", REGISTRY) is None


def test_a_geocoder_failure_never_raises(monkeypatch, instant):
    """A geocoder is a garnish on a dispatch hint. The request still has a
    transcript, a classification and an incident record."""
    def boom(*a, **k):
        raise OSError("connection reset")

    monkeypatch.setenv("GEOCODER", "nominatim")
    monkeypatch.setattr(geocode.urllib.request, "urlopen", boom)
    assert locate("nowhere in the table", REGISTRY) is None


def test_a_repeated_address_is_not_looked_up_twice(monkeypatch, instant):
    calls = []

    def once(*a, **k):
        calls.append(1)
        return _Resp([{"lat": "1", "lon": "2"}])

    monkeypatch.setenv("GEOCODER", "nominatim")
    monkeypatch.setattr(geocode.urllib.request, "urlopen", once)
    locate("nowhere in the table", REGISTRY)
    locate("Nowhere In The Table.", REGISTRY)      # same place, spoken twice
    assert len(calls) == 1
