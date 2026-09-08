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
    "eta_min": 15,
    "units": ["emergency_team"],
}


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
    """Pick a station for an emergency type, preferring an area keyword match.

    Falls back to the first station listed for the type, then to a placeholder,
    so this always returns the same shape.
    """
    registry = load_stations() if stations is None else stations
    candidates = registry.get(emergency_type) or registry.get("accident") or []
    if not candidates:
        return dict(FALLBACK)

    location = (probable_location or "").lower()
    chosen = next(
        (s for s in candidates
         if any(k in location for k in s.get("area_keywords", []))),
        candidates[0],
    )
    return {
        "station": chosen.get("name", FALLBACK["station"]),
        "eta_min": chosen.get("base_eta_min", FALLBACK["eta_min"]),
        "units": chosen.get("units", list(FALLBACK["units"])),
    }
