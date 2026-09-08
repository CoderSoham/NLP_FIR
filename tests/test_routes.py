"""Route and template behaviour, without the ML stack loaded."""
import pytest

from conftest import upload


def test_upload_renders_only_this_jobs_plots(pipeline):
    html = upload(pipeline.client).get_data(as_text=True)
    assert html.count(f"/plot/{pipeline.report_id}/") == 4


def test_plot_urls_are_job_scoped_not_shared_filenames(pipeline):
    """Regression: plots were once fixed names on the public static mount."""
    html = upload(pipeline.client).get_data(as_text=True)
    assert "plots/waveform.png" not in html
    assert "static/plots" not in html


def test_job_without_entities_renders_no_entity_image(pipeline):
    """Regression: the entity plot was referenced but never generated."""
    pipeline.plots = ["waveform", "mfcc", "pitch"]
    html = upload(pipeline.client).get_data(as_text=True)
    assert html.count(f"/plot/{pipeline.report_id}/") == 3
    assert "Entities Distribution" not in html


def test_plot_serves_for_its_own_job(pipeline):
    upload(pipeline.client)
    r = pipeline.client.get(f"/plot/{pipeline.report_id}/waveform")
    assert r.status_code == 200
    assert r.mimetype == "image/png"


@pytest.mark.parametrize("kind", ["passwd", "config", "waveform.png", "../pitch"])
def test_unknown_plot_kind_is_rejected(pipeline, kind):
    upload(pipeline.client)
    assert pipeline.client.get(f"/plot/{pipeline.report_id}/{kind}").status_code == 404


@pytest.mark.parametrize("report_id", ["NOTHEX", "../../config", "a" * 31, "a" * 33])
def test_malformed_report_id_is_rejected(pipeline, report_id):
    """404 rather than 500 -- both path segments are user-controlled."""
    assert pipeline.client.get(f"/plot/{report_id}/waveform").status_code == 404


def test_absent_plot_is_404_not_500(pipeline):
    absent = "b" * 32
    assert pipeline.client.get(f"/plot/{absent}/waveform").status_code == 404


def test_audio_route_rejects_names_it_did_not_generate(pipeline):
    (pipeline.uploads / "secret.mp3").write_bytes(b"ID3")
    assert pipeline.client.get("/audio/secret.mp3").status_code == 404


def test_audio_route_serves_generated_names(pipeline):
    name = f"{'c' * 32}.mp3"
    (pipeline.uploads / name).write_bytes(b"ID3")
    assert pipeline.client.get(f"/audio/{name}").status_code == 200


def test_rejects_unsupported_extension(pipeline):
    r = upload(pipeline.client, name="payload.exe")
    assert r.status_code == 400


def test_healthz(pipeline):
    assert pipeline.client.get("/healthz").get_json() == {"status": "ok"}


def test_api_returns_the_analysis_as_json(pipeline):
    import io
    r = pipeline.client.post(
        "/api/process",
        data={"audio_file": (io.BytesIO(b"x"), "call.mp3")},
        content_type="multipart/form-data",
    )
    body = r.get_json()
    assert r.status_code == 200
    assert body["report_id"] == pipeline.report_id
    assert body["severity"] == "high"
    assert body["plots"] == ["waveform", "mfcc", "pitch", "entities"]
