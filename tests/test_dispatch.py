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
    assert got["basis"] == "location_match"
    assert got["matched_on"] == "east"
    # A keyword match knows which station, not how far. The number is the
    # station's nominal response time and says so.
    assert got["eta_basis"] == "registry_constant"


def test_falls_back_to_first_station_when_location_is_unknown():
    got = get_dispatch_suggestion("fire", "nowhere", stations=REGISTRY)
    assert got["station"] == "Station 1"
    assert got["basis"] == "default_for_type"


def test_missing_location_does_not_raise():
    assert get_dispatch_suggestion("fire", None, stations=REGISTRY)["station"] == "Station 1"


def test_no_location_withholds_the_eta():
    """Regression: a real call had location=None and still got 'ETA 4 min'.

    A stored constant rendered as an arrival time is indistinguishable from a
    computed one, so it is withheld unless the station was actually matched.
    """
    got = get_dispatch_suggestion("fire", None, stations=REGISTRY)
    assert got["eta_min"] is None
    assert got["basis"] == "default_for_type"
    assert got["matched_on"] is None


def test_unmatched_location_also_withholds_the_eta():
    got = get_dispatch_suggestion("fire", "Atlantis", stations=REGISTRY)
    assert got["eta_min"] is None


def test_every_result_reports_a_basis():
    for loc in ("East Alameda", "nowhere", None, ""):
        assert get_dispatch_suggestion("fire", loc, stations=REGISTRY)["basis"] in (
            "location_match", "default_for_type", "no_registry")


def test_unknown_type_routes_to_accident():
    assert get_dispatch_suggestion("volcano", "highway 9", stations=REGISTRY)["station"] == "Traffic Response"


def test_empty_registry_returns_the_fallback_shape():
    got = get_dispatch_suggestion("fire", "downtown", stations={})
    assert got == FALLBACK
    assert set(got) == {"station", "eta_min", "units", "basis", "matched_on",
                        "distance_km", "eta_basis", "example_data"}
    assert got["eta_min"] is None


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
    from utils.dispatch import station_types

    stations = load_stations(force=True)
    assert stations, "stations.json should load"
    # station_types, not .items(): the registry also carries metadata and a
    # geocoding table, and iterating those as station lists is what broke
    # this test when the example marker was added.
    for kind in station_types(stations):
        for entry in stations[kind]:
            assert {"name", "base_eta_min", "units"} <= set(entry), kind


# ---- ISSUE-023: a computed distance, and an honest label on invented data ---

import math

from utils.dispatch import (ETA_COMPUTED, ETA_CONSTANT, coordinates,
                            eta_minutes, haversine_km, is_example,
                            station_types)

# Two real Kansas City coordinates, used only as arithmetic fixtures.
DOWNTOWN = (39.0997, -94.5786)
BANNISTER = (38.9339, -94.5641)          # ~18.5 km south

PLACED = {
    "police": [
        {"name": "North Patrol", "lat": 39.1800, "lon": -94.5780,
         "units": ["patrol"]},
        {"name": "South Patrol", "lat": 38.9400, "lon": -94.5650,
         "units": ["patrol"]},
    ],
}


def test_haversine_matches_a_known_distance():
    assert math.isclose(haversine_km(DOWNTOWN, BANNISTER), 18.4, abs_tol=0.5)


def test_haversine_is_zero_for_the_same_point():
    assert haversine_km(DOWNTOWN, DOWNTOWN) == 0.0


def test_haversine_is_symmetric():
    assert math.isclose(haversine_km(DOWNTOWN, BANNISTER),
                        haversine_km(BANNISTER, DOWNTOWN))


def test_the_nearest_station_wins_on_distance_not_order():
    got = get_dispatch_suggestion("police", None, stations=PLACED,
                                  incident_coords=BANNISTER)
    assert got["station"] == "South Patrol"
    assert got["basis"] == "nearest_by_distance"
    assert got["eta_basis"] == ETA_COMPUTED
    assert got["distance_km"] < 1.0


def test_a_computed_eta_accounts_for_roads_not_crow_flight():
    """Straight-line distance understates road distance, so the detour factor
    is applied before dividing by speed."""
    direct = 18.4 / 50.0 * 60                 # ~22 minutes as the crow flies
    assert eta_minutes(18.4, speed_kmh=50) > direct


def test_an_eta_is_rounded_up_never_down():
    """Arriving before the estimate is a surprise; after it is a broken
    promise made to someone waiting for an ambulance."""
    assert eta_minutes(0.1, speed_kmh=50) == 1
    assert eta_minutes(10.0, speed_kmh=50) == math.ceil(10 * 1.3 / 50 * 60)


@pytest.mark.parametrize("speed", [0, -10])
def test_a_nonsense_speed_withholds_the_eta(speed):
    assert eta_minutes(5.0, speed_kmh=speed) is None


def test_speed_is_configurable(monkeypatch):
    monkeypatch.setenv("DISPATCH_SPEED_KMH", "100")
    fast = eta_minutes(20.0)
    monkeypatch.setenv("DISPATCH_SPEED_KMH", "25")
    assert eta_minutes(20.0) > fast


def test_distance_is_skipped_when_no_station_has_coordinates():
    """Falls back to keywords rather than inventing a position."""
    got = get_dispatch_suggestion("fire", "East Alameda", stations=REGISTRY,
                                  incident_coords=BANNISTER)
    assert got["basis"] == "location_match"
    assert got["distance_km"] is None


@pytest.mark.parametrize("entry", [
    {}, None, {"lat": 1}, {"lat": "x", "lon": "y"}, {"lat": None, "lon": None},
    {"lat": 91, "lon": 0},                    # beyond the poles
    {"lat": 0, "lon": 181},                   # beyond the antimeridian
])
def test_unusable_coordinates_are_rejected_not_guessed(entry):
    assert coordinates(entry) is None


def test_swapped_latitude_and_longitude_is_caught_when_out_of_range():
    """The most common error in a station CSV, and it silently relocates a
    station to another hemisphere."""
    assert coordinates({"lat": -94.5786, "lon": 39.0997}) is None


def test_valid_coordinates_survive_as_floats():
    assert coordinates({"lat": "39.1", "lon": "-94.6"}) == (39.1, -94.6)


# ---- the example marker ----------------------------------------------------

def test_the_shipped_registry_admits_it_is_example_data():
    """ISSUE-023. Every surface that names a station reads this flag."""
    assert is_example(load_stations(force=True)) is True


def test_a_registry_without_the_marker_is_not_example_data():
    assert is_example(REGISTRY) is False
    assert get_dispatch_suggestion("fire", "downtown",
                                   stations=REGISTRY)["example_data"] is False


def test_the_example_flag_reaches_every_outcome():
    marked = dict(REGISTRY, _example=True)
    for location in ("East Alameda", "nowhere", None):
        got = get_dispatch_suggestion("fire", location, stations=marked)
        assert got["example_data"] is True
    assert get_dispatch_suggestion(
        "fire", "x", stations={"_example": True})["example_data"] is True


def test_metadata_keys_are_not_mistaken_for_station_lists():
    registry = {"_example": True, "_comment": "x", "places": {"a": {}},
                "fire": REGISTRY["fire"]}
    assert station_types(registry) == ["fire"]
    assert get_dispatch_suggestion("places", "downtown",
                                   stations=registry)["station"] == "Unassigned"


def test_a_keyword_match_labels_its_eta_as_a_stored_constant():
    got = get_dispatch_suggestion("fire", "East Alameda", stations=REGISTRY)
    assert got["eta_min"] == 9 and got["eta_basis"] == ETA_CONSTANT


def test_a_station_with_no_stored_eta_reports_no_eta_basis():
    registry = {"fire": [{"name": "S", "area_keywords": ["downtown"]}]}
    got = get_dispatch_suggestion("fire", "downtown", stations=registry)
    assert got["eta_min"] is None and got["eta_basis"] is None
