"""The tool that turns a real station CSV into a registry.

The point of ISSUE-023 is that the shipped registry is invented. This is how
someone replaces it, so its failure modes matter: a wrong coordinate produces
a confident ETA to the wrong place, which is worse than no ETA.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from build_stations import build, main, read_rows, split_list

GOOD = """name,type,lat,lon,units,area_keywords
Station 19,fire,39.0421,-94.5661,fire_truck|ladder,bannister|hickman mills
Truman Medical,medical,39.0862,-94.5679,ambulance,downtown
"""


def write(tmp_path, text, name="stations.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_a_good_csv_becomes_a_registry(tmp_path):
    registry, skipped = build(read_rows(write(tmp_path, GOOD)))
    assert not skipped
    assert set(registry) == {"fire", "medical"}
    station = registry["fire"][0]
    assert station["name"] == "Station 19"
    assert (station["lat"], station["lon"]) == (39.0421, -94.5661)
    assert station["units"] == ["fire_truck", "ladder"]
    assert station["area_keywords"] == ["bannister", "hickman mills"]


def test_swapped_latitude_and_longitude_is_skipped_with_a_reason(tmp_path):
    """The single most common error in this kind of file. It relocates a
    station to the wrong hemisphere without anything complaining."""
    csv = "name,type,lat,lon\nS,fire,-94.5661,39.0421\n"
    registry, skipped = build(read_rows(write(tmp_path, csv)))
    assert registry == {}
    assert len(skipped) == 1 and "out of range" in skipped[0][2]


@pytest.mark.parametrize("row,reason", [
    ("S,fire,notanumber,0", "not numeric"),
    ("S,fire,,", "not numeric"),
    (",fire,1,2", "empty"),
    ("S,,1,2", "empty"),
])
def test_unusable_rows_are_skipped_not_guessed(tmp_path, row, reason):
    registry, skipped = build(read_rows(write(
        tmp_path, "name,type,lat,lon\n" + row + "\n")))
    assert registry == {} and len(skipped) == 1


def test_a_skipped_row_reports_its_line_number(tmp_path):
    """Row 1 is the header, so the first data row is row 2 -- which is what
    a spreadsheet shows the person fixing it."""
    csv = "name,type,lat,lon\nGood,fire,39.0,-94.5\nBad,fire,x,y\n"
    _registry, skipped = build(read_rows(write(tmp_path, csv)))
    assert skipped[0][0] == 3


def test_a_missing_required_column_fails_loudly(tmp_path):
    with pytest.raises(SystemExit, match="missing required column"):
        read_rows(write(tmp_path, "name,type,latitude,longitude\nS,fire,1,2\n"))


def test_a_utf8_bom_does_not_break_the_header(tmp_path):
    """Excel writes one, and it turns the first column name into '\\ufeffname'."""
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbf" + GOOD.encode())
    assert len(read_rows(str(path))) == 2


def test_rows_with_no_units_get_the_default(tmp_path):
    registry, _ = build(read_rows(write(
        tmp_path, "name,type,lat,lon\nS,fire,39.0,-94.5\n")), ["emergency_team"])
    assert registry["fire"][0]["units"] == ["emergency_team"]


def test_the_output_has_no_example_marker(tmp_path, capsys):
    """A registry built from real data is not example data, and the
    placeholder label in the UI disappears with it."""
    out = tmp_path / "out.json"
    main([write(tmp_path, GOOD), "--out", str(out)])
    written = json.load(open(out))
    assert "_example" not in written
    from utils.dispatch import is_example
    assert is_example(written) is False


def test_the_result_loads_as_a_registry(tmp_path):
    from utils.dispatch import get_dispatch_suggestion, load_stations

    out = tmp_path / "out.json"
    main([write(tmp_path, GOOD), "--out", str(out)])
    registry = load_stations(path=str(out), force=True)
    got = get_dispatch_suggestion("fire", None, stations=registry,
                                  incident_coords=(39.0421, -94.5661))
    assert got["station"] == "Station 19"
    assert got["basis"] == "nearest_by_distance"
    assert got["eta_min"] == 1               # it is at the incident
    assert got["example_data"] is False
    load_stations(force=True)                # restore the shipped registry


def test_a_csv_with_no_usable_rows_writes_nothing(tmp_path):
    with pytest.raises(SystemExit, match="no usable rows"):
        main([write(tmp_path, "name,type,lat,lon\nS,fire,x,y\n"),
              "--out", str(tmp_path / "never.json")])
    assert not (tmp_path / "never.json").exists()


@pytest.mark.parametrize("value,expected", [
    ("a|b", ["a", "b"]), (" a | b ", ["a", "b"]), ("", []), (None, []),
    ("a||b", ["a", "b"]),
])
def test_pipe_separated_fields(value, expected):
    assert split_list(value) == expected
