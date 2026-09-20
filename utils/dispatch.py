"""Station registry and dispatch suggestion.

The registry is data, not code: `stations.json` is read at first use and can be
replaced with your own stations without touching the source. The file that
ships is a small example set — swap it for a real one via `STATIONS_FILE`.

No ML imports here, so it is testable on its own.
"""
import json
import os

_STATIONS = None

DEFAULT_STATIONS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stations.json"
)

FALLBACK = {
    "station": "Unassigned",
    "eta_min": None,
    "units": ["emergency_team"],
    "basis": "no_registry",
    "matched_on": None,
}

# How the station was chosen. Reported so a reader can tell a real match from a
# default -- with no location, this used to return the first station in the list
# and an ETA, indistinguishable from a genuine match.
BASIS_LOCATION = "location_match"
BASIS_DEFAULT = "default_for_type"
BASIS_NONE = "no_registry"


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


def get_dispatch_suggestion(emergency_type, probable_location, stations=None):
    """Pick a station for an emergency type, reporting how it was chosen.

    `basis` distinguishes a real area-keyword match from the fallback. Without
    it the caller cannot tell whether "Precinct A, ETA 4 min" came from the
    caller's address or from the order of a list -- on a real sample call the
    location was None and this still returned a station and a confident ETA.

    `eta_min` is None unless the station was matched on location. A stored
    constant presented as an arrival time is the most misleading field the
    pipeline can emit, so it is withheld rather than guessed.
    """
    registry = load_stations() if stations is None else stations
    candidates = registry.get(emergency_type) or registry.get("accident") or []
    if not candidates:
        return dict(FALLBACK)

    location = (probable_location or "").lower()
    matched_on = None
    chosen = None
    if location:
        for station in candidates:
            hit = next((k for k in station.get("area_keywords", []) if k in location), None)
            if hit:
                chosen, matched_on = station, hit
                break

    if chosen is None:
        chosen = candidates[0]
        return {
            "station": chosen.get("name", FALLBACK["station"]),
            "eta_min": None,
            "units": chosen.get("units", list(FALLBACK["units"])),
            "basis": BASIS_DEFAULT,
            "matched_on": None,
        }

    return {
        "station": chosen.get("name", FALLBACK["station"]),
        "eta_min": chosen.get("base_eta_min"),
        "units": chosen.get("units", list(FALLBACK["units"])),
        "basis": BASIS_LOCATION,
        "matched_on": matched_on,
    }
