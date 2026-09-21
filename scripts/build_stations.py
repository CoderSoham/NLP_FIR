#!/usr/bin/env python3
"""Build a station registry from a CSV of real stations.

The registry that ships is example data and says so. This turns a list of
real stations into one the pipeline can compute distances against:

    python scripts/build_stations.py stations.csv --out my-stations.json
    STATIONS_FILE=my-stations.json GEOCODER=registry python app.py

The CSV needs a header with at least `name`, `type`, `lat`, `lon`:

    name,type,lat,lon,units,area_keywords
    Station 19,fire,39.0421,-94.5661,fire_truck|ladder,bannister|hickman mills
    Truman Medical,medical,39.0862,-94.5679,ambulance,downtown

`units` and `area_keywords` are pipe-separated and optional. `type` is one of
medical, fire, police, accident -- anything else is kept and used only when a
call is classified that way.

Most cities publish this. Search for "<city> fire station locations open data";
the usual formats are CSV, GeoJSON or ArcGIS, and all three export to CSV. Use
the authority's own list rather than a scraped one: a wrong coordinate here
produces a confident ETA to the wrong place, which is worse than no ETA.
"""
import argparse
import csv
import json
import sys

REQUIRED = ("name", "type", "lat", "lon")
KNOWN_TYPES = ("medical", "fire", "police", "accident")


def split_list(value):
    return [part.strip() for part in (value or "").split("|") if part.strip()]


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                f"{path}: missing required column(s): {', '.join(missing)}\n"
                f"found: {', '.join(reader.fieldnames or ['nothing'])}")
        return list(reader)


def build(rows, default_units=None):
    registry, skipped = {}, []
    for number, row in enumerate(rows, start=2):      # row 1 is the header
        name, kind = (row.get("name") or "").strip(), (row.get("type") or "").strip().lower()
        try:
            lat, lon = float(row["lat"]), float(row["lon"])
        except (TypeError, ValueError):
            skipped.append((number, name or "?", "lat/lon not numeric"))
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            # Swapped lat and lon is the single most common error in this
            # kind of file, and it puts the station in the wrong hemisphere
            # without anything complaining.
            skipped.append((number, name or "?", f"out of range ({lat}, {lon})"))
            continue
        if not name or not kind:
            skipped.append((number, name or "?", "name or type is empty"))
            continue

        station = {"name": name, "lat": lat, "lon": lon,
                   "area_keywords": [k.lower() for k in
                                     split_list(row.get("area_keywords"))],
                   "units": split_list(row.get("units")) or list(default_units or [])}
        registry.setdefault(kind, []).append(station)
    return registry, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV of real stations")
    ap.add_argument("--out", default="stations.local.json")
    ap.add_argument("--units", default="emergency_team",
                    help="pipe-separated units for rows that name none")
    args = ap.parse_args(argv)

    registry, skipped = build(read_rows(args.csv), split_list(args.units))
    if not registry:
        raise SystemExit("no usable rows; nothing written")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)

    total = sum(len(v) for v in registry.values())
    print(f"wrote {args.out}: {total} stations across "
          f"{', '.join(sorted(registry))}")
    for kind in sorted(registry):
        if kind not in KNOWN_TYPES:
            print(f"  note: type {kind!r} is not one the classifier emits "
                  f"({', '.join(KNOWN_TYPES)}), so it will never be selected")
    for number, name, why in skipped:
        print(f"  skipped row {number} ({name}): {why}", file=sys.stderr)
    # No _example key: a registry built from real data is not example data,
    # and the placeholder label in the UI disappears with it.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
