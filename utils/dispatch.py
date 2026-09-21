"""Station registry and dispatch suggestion.

The registry is data, not code: `stations.json` is read at first use and can
be replaced with your own via `STATIONS_FILE`.

Two things a reader needs to be able to tell apart, and which this module
exists to keep separate:

**Is the answer computed or looked up?** With coordinates on the station and
on the incident, the ETA is a distance divided by a speed, and the distance is
real. Without them it is a stored constant that someone typed, which is the
most misleading number this pipeline can emit -- so it is withheld rather than
shown, and `basis` says which happened.

**Is the registry real?** The file that ships is a seven-station example with
invented names and no coordinates. It declares itself with `"_example": true`,
and every surface that displays a station says so. A single fabricated field
on a page where everything else is evidenced is worse than the same field on a
rough one, because the page has earned enough trust to be believed.

No ML imports here, so it is testable on its own.
"""
import json
import math
import os

_STATIONS = None

DEFAULT_STATIONS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stations.json"
)

# Average emergency-vehicle speed for the straight-line estimate, km/h. Road
# distance exceeds straight-line distance, so DETOUR_FACTOR converts one to
# the other -- 1.3 is the usual planning figure for urban street grids.
# Neither is a routing engine, and the output says "estimate" for that reason.
DEFAULT_SPEED_KMH = 50.0
DETOUR_FACTOR = 1.3
EARTH_RADIUS_KM = 6371.0

FALLBACK = {
    "station": "Unassigned",
    "eta_min": None,
    "units": ["emergency_team"],
    "basis": "no_registry",
    "matched_on": None,
    "distance_km": None,
    "eta_basis": None,
    "example_data": False,
}

# Where the ETA came from. The number alone cannot carry this, and the
# difference is the whole point: one is a distance divided by a speed, the
# other is a figure someone typed into a JSON file for that station in
# general, with no knowledge of where this incident is.
ETA_COMPUTED = "computed_from_distance"
ETA_CONSTANT = "registry_constant"

# How the station was chosen. Reported so a reader can tell a real match from
# a default -- with no location, this used to return the first station in the
# list and an ETA, indistinguishable from a genuine match.
BASIS_DISTANCE = "nearest_by_distance"
BASIS_LOCATION = "location_match"
BASIS_DEFAULT = "default_for_type"
BASIS_NONE = "no_registry"


def haversine_km(a, b):
    """Great-circle distance between two (lat, lon) pairs, in kilometres."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, h)))


def eta_minutes(distance_km, speed_kmh=None):
    """Minutes to cover a straight-line distance by road, rounded up.

    Rounded up, never down: an ETA that arrives before the estimate is a
    pleasant surprise, and one that arrives after it is a broken promise made
    to someone waiting for an ambulance.
    """
    # `is None`, not `or`: an explicit speed of 0 is nonsense the caller
    # supplied and should be rejected, not quietly replaced by the default.
    if speed_kmh is None:
        try:
            speed_kmh = float(os.environ.get("DISPATCH_SPEED_KMH",
                                             DEFAULT_SPEED_KMH))
        except ValueError:
            speed_kmh = DEFAULT_SPEED_KMH
    if distance_km is None or speed_kmh <= 0:
        return None
    return max(1, math.ceil(distance_km * DETOUR_FACTOR / speed_kmh * 60))


def coordinates(entry):
    """(lat, lon) from a mapping, or None when either is missing or unusable."""
    if not isinstance(entry, dict):
        return None
    try:
        lat, lon = float(entry["lat"]), float(entry["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    # Out-of-range values are a data error, not a location. Returning them
    # would produce a confident distance to somewhere that does not exist.
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def load_stations(path=None, force=False):
    """Read the registry once and cache it.

    A missing or malformed file yields an empty registry rather than raising:
    dispatch is a hint alongside the analysis, and losing it should not fail a
    request that has already transcribed and classified a call.
    """
    global _STATIONS
    if _STATIONS is not None and not force:
        return _STATIONS
    path = path or os.environ.get("STATIONS_FILE") or DEFAULT_STATIONS_FILE
    try:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        _STATIONS = loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError):
        _STATIONS = {}
    return _STATIONS


def is_example(registry):
    """Does this registry admit to being example data?"""
    return bool((registry or {}).get("_example"))


# Keys that are registry metadata rather than an emergency type. A registry
# is a mapping of type to station list plus these; without the distinction,
# `get_dispatch_suggestion("places", ...)` would treat the geocoding table as
# a list of stations and quietly return nothing.
RESERVED_KEYS = ("places",)


def station_types(registry):
    """Emergency types this registry has stations for."""
    return [key for key, value in (registry or {}).items()
            if isinstance(value, list) and not key.startswith("_")
            and key not in RESERVED_KEYS]


def _station_list(registry, emergency_type):
    types = station_types(registry)
    key = emergency_type if emergency_type in types else (
        "accident" if "accident" in types else None)
    if key is None:
        return []
    return [s for s in registry[key] if isinstance(s, dict)]


def _result(station, basis, registry, matched_on=None, distance_km=None,
            eta_min=None, eta_basis=None):
    return {
        "station": station.get("name", FALLBACK["station"]),
        "eta_min": eta_min,
        "units": station.get("units", list(FALLBACK["units"])),
        "basis": basis,
        "matched_on": matched_on,
        "distance_km": (round(distance_km, 1)
                        if distance_km is not None else None),
        "eta_basis": eta_basis,
        "example_data": is_example(registry),
    }


def get_dispatch_suggestion(emergency_type, probable_location, stations=None,
                            incident_coords=None):
    """Pick a station for an emergency type, reporting how it was chosen.

    Three strategies, best first:

    1. **Distance.** Both the incident and the station have coordinates, so
       the nearest one wins and the ETA is computed from the distance.
    2. **Area keyword.** The extracted location string contains a keyword the
       station claims. There is no distance, so any ETA can only be the
       station's stored `base_eta_min` -- its nominal response time, not an
       estimate for this incident. It is reported with
       `eta_basis="registry_constant"` so the display can say which it is.
    3. **First in the list.** Nothing matched. Reported as
       `default_for_type` precisely so it cannot be mistaken for the others.
    """
    registry = load_stations() if stations is None else stations
    candidates = _station_list(registry, emergency_type)
    if not candidates:
        return dict(FALLBACK, example_data=is_example(registry))

    if incident_coords:
        placed = [(haversine_km(incident_coords, c), s)
                  for s in candidates
                  for c in [coordinates(s)] if c]
        if placed:
            distance, station = min(placed, key=lambda pair: pair[0])
            return _result(station, BASIS_DISTANCE, registry,
                           distance_km=distance,
                           eta_min=eta_minutes(distance),
                           eta_basis=ETA_COMPUTED)

    location = (probable_location or "").lower()
    if location:
        for station in candidates:
            hit = next((k for k in station.get("area_keywords", [])
                        if k and k in location), None)
            if hit:
                nominal = station.get("base_eta_min")
                return _result(
                    station, BASIS_LOCATION, registry, matched_on=hit,
                    eta_min=nominal,
                    eta_basis=ETA_CONSTANT if nominal is not None else None)

    return _result(candidates[0], BASIS_DEFAULT, registry)
