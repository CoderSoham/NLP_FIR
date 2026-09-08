"""Entity-distribution plot. Imports without torch, so it runs in the fast suite."""
import os

import pytest

from utils.plots import generate_entity_plot, plot_path

RID = "a" * 32


def ents(*labels):
    return [{"text": f"t{i}", "label": l} for i, l in enumerate(labels)]


def test_writes_a_png_keyed_by_report_id(tmp_path):
    assert generate_entity_plot(ents("LOC", "PERSON"), str(tmp_path), RID) is True
    written = plot_path(str(tmp_path), "entities", RID)
    assert os.path.isfile(written)
    assert os.path.getsize(written) > 1000
    with open(written, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"


def test_no_entities_writes_nothing(tmp_path):
    """Regression: the template referenced a plot nothing generated.

    spaCy finding no entities is normal, so this returns False and the caller
    omits the image rather than rendering an empty figure or a broken one.
    """
    assert generate_entity_plot([], str(tmp_path), RID) is False
    assert os.listdir(tmp_path) == []


def test_two_jobs_do_not_share_a_file(tmp_path):
    """Regression: plots were once fixed filenames shared by every request."""
    other = "b" * 32
    generate_entity_plot(ents("LOC"), str(tmp_path), RID)
    generate_entity_plot(ents("PERSON", "ORG", "ORG"), str(tmp_path), other)
    assert os.path.isfile(plot_path(str(tmp_path), "entities", RID))
    assert os.path.isfile(plot_path(str(tmp_path), "entities", other))
    assert len(os.listdir(tmp_path)) == 2


@pytest.mark.parametrize("labels", [("WORK_OF_ART",), ("LOC",) * 40, ("A", "B", "C")])
def test_label_shapes_render(tmp_path, labels):
    assert generate_entity_plot(ents(*labels), str(tmp_path), RID) is True


def test_creates_the_output_folder(tmp_path):
    nested = tmp_path / "does" / "not" / "exist"
    assert generate_entity_plot(ents("LOC"), str(nested), RID) is True
    assert nested.is_dir()
