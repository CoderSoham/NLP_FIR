"""Turning a spoken address into coordinates, when you have asked for it.

Off by default, and the default matters more than the feature. A 911
transcript's location field is the most identifying string the pipeline
produces -- "1331 East Bannister Road" plus a timestamp plus a shooting is not
anonymous. Sending it to a third-party geocoder on every request, silently,
because it makes an ETA look better, is not a trade this project gets to make
on an operator's behalf.

So: `GEOCODER=nominatim` turns it on, nothing turns it on by itself, and the
result page names the geocoder that was used.

Two backends:

- `registry` -- an offline table of place name to coordinate, supplied in the
  stations file. No network, no third party, and exact for the places you
  actually cover. This is the one to use in production.
- `nominatim` -- OpenStreetMap's public geocoder. Free, no key, rate limited
  to one request a second, and it receives the address.
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request

from utils.dispatch import coordinates

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim's usage policy requires a real identifying User-Agent and at most
# one request per second. Both are conditions of the free service, not
# suggestions, and ignoring them gets the whole project blocked.
USER_AGENT = os.environ.get(
    "GEOCODER_USER_AGENT",
    "nlp-fir/1.0 (emergency call triage; https://github.com/CoderSoham/NLP_FIR)")
MIN_INTERVAL_S = 1.0

_LAST_CALL = [0.0]
_CACHE = {}


def mode():
    """`GEOCODER`: off | registry | nominatim. Unknown values mean off."""
    chosen = os.environ.get("GEOCODER", "off").strip().lower()
    return chosen if chosen in ("off", "registry", "nominatim") else "off"


def normalise(text):
    """A cache key that survives transcription noise in the same address."""
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def from_registry(location, registry):
    """Look the place up in the stations file's own `places` table.

    Longest key first, so "east bannister" beats "bannister" when both are
    listed -- the more specific entry is the one someone bothered to add.
    """
    places = (registry or {}).get("places") or {}
    text = normalise(location)
    if not text:
        return None
    for key in sorted(places, key=len, reverse=True):
        if normalise(key) and normalise(key) in text:
            return coordinates(places[key])
    return None


def from_nominatim(location, timeout=10):
    """Ask OpenStreetMap. Returns None on any failure, never raises.

    A geocoder is a garnish on a dispatch hint. If it is down, slow, or has
    never heard of the street, the request still has a transcript, a
    classification and an incident record, and none of those should be put at
    risk for an ETA.
    """
    text = (location or "").strip()
    if not text:
        return None
    key = normalise(text)
    if key in _CACHE:
        return _CACHE[key]

    wait = MIN_INTERVAL_S - (time.monotonic() - _LAST_CALL[0])
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL[0] = time.monotonic()

    url = NOMINATIM_URL + "?" + urllib.parse.urlencode(
        {"q": text, "format": "json", "limit": 1})
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            results = json.loads(response.read().decode())
        found = coordinates(results[0]) if results else None
    except Exception:
        found = None
    _CACHE[key] = found
    return found


def locate(location, registry=None):
    """(lat, lon) for a location string, or None. Honours `GEOCODER`.

    The registry table is consulted first whatever the mode, because it costs
    nothing, sends nothing anywhere, and is more accurate than a public
    geocoder for the places an operator has bothered to list.
    """
    chosen = mode()
    if chosen == "off" or not (location or "").strip():
        return None
    found = from_registry(location, registry)
    if found or chosen == "registry":
        return found
    return from_nominatim(location)
