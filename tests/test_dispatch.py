"""Station registry and dispatch selection. No models, no app."""
import json

import pytest

from utils.dispatch import FALLBACK, get_dispatch_suggestion, load_stations

REGISTRY = {
    "fire": [
        {"name": "Station 1", "area_keywords": ["downtown"], "base_eta_min": 5,
         "units": ["fire_truck"]},
        {"name": "Station 7", "area_keywords": ["east"], "base_eta_min": 9,
         "units": ["fire_truck"]},
    ],
    "accident": [
        {"name": "Traffic Response", "area_keywords": ["highway"],
         "base_eta_min": 7, "units": ["traffic_unit"]},
    ],
}


def test_keyword_match_wins_over_order():
    got = get_dispatch_suggestion("fire", "East Alameda", stations=REGISTRY)
    assert got["station"] == "Station 7"
    assert got["eta_min"] == 9


def test_falls_back_to_first_station_when_location_is_unknown():
    assert get_dispatch_suggestion("fire", "nowhere", stations=REGISTRY)["station"] == "Station 1"


def test_missing_location_does_not_raise():
    assert get_dispatch_suggestion("fire", None, stations=REGISTRY)["station"] == "Station 1"


def test_unknown_type_routes_to_accident():
    assert get_dispatch_suggestion("volcano", "highway 9", stations=REGISTRY)["station"] == "Traffic Response"


def test_empty_registry_returns_the_fallback_shape():
    got = get_dispatch_suggestion("fire", "downtown", stations={})
    assert got == FALLBACK
    assert set(got) == {"station", "eta_min", "units"}


def test_registry_loads_from_a_file(tmp_path):
    path = tmp_path / "stations.json"
    path.write_text(json.dumps(REGISTRY))
    assert load_stations(str(path), force=True)["fire"][0]["name"] == "Station 1"


@pytest.mark.parametrize("content", ["not json", "[]", ""])
def test_malformed_registry_degrades_instead_of_raising(tmp_path, content):
    """Dispatch is a hint; losing it must not fail an already-analysed call."""
    path = tmp_path / "bad.json"
    path.write_text(content)
    assert load_stations(str(path), force=True) == {}


def test_missing_file_degrades(tmp_path):
    assert load_stations(str(tmp_path / "absent.json"), force=True) == {}


def test_shipped_registry_is_valid():
    stations = load_stations(force=True)
    assert stations, "stations.json should load"
    for kind, entries in stations.items():
        for entry in entries:
            assert {"name", "base_eta_min", "units"} <= set(entry), kind
