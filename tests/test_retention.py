"""Retention sweeps. Pure filesystem, no app and no models needed."""
import os
import time

from utils.retention import sweep


def touch(path, age_seconds=0):
    path.write_bytes(b"x")
    if age_seconds:
        old = time.time() - age_seconds
        os.utime(path, (old, old))
    return path


def test_deletes_files_past_the_cutoff(tmp_path):
    old = touch(tmp_path / "old.mp3", age_seconds=7200)
    assert sweep(str(tmp_path), max_age_seconds=3600) == 1
    assert not old.exists()


def test_keeps_files_inside_the_window(tmp_path):
    fresh = touch(tmp_path / "fresh.mp3", age_seconds=60)
    assert sweep(str(tmp_path), max_age_seconds=3600) == 0
    assert fresh.exists()


def test_sweeps_several_folders(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    touch(a / "1.mp3", 7200)
    touch(b / "2.pdf", 7200)
    touch(b / "3.png", 10)
    assert sweep(str(a), str(b), max_age_seconds=3600) == 2
    assert (b / "3.png").exists()


def test_missing_folder_is_not_an_error(tmp_path):
    assert sweep(str(tmp_path / "nope"), None, max_age_seconds=1) == 0


def test_leaves_subdirectories_alone(tmp_path):
    nested = tmp_path / "sub"
    nested.mkdir()
    old = time.time() - 7200
    os.utime(nested, (old, old))
    assert sweep(str(tmp_path), max_age_seconds=3600) == 0
    assert nested.is_dir()


def test_is_idempotent(tmp_path):
    touch(tmp_path / "old.mp3", 7200)
    assert sweep(str(tmp_path), max_age_seconds=3600) == 1
    assert sweep(str(tmp_path), max_age_seconds=3600) == 0


def test_boundary_uses_injected_clock(tmp_path):
    """now is injectable so the boundary is testable without sleeping."""
    p = touch(tmp_path / "x.mp3")
    mtime = os.path.getmtime(p)
    assert sweep(str(tmp_path), max_age_seconds=10, now=mtime + 9) == 0
    assert sweep(str(tmp_path), max_age_seconds=10, now=mtime + 11) == 1
