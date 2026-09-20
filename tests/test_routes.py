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


def test_upload_returns_a_progress_page_not_a_result(pipeline):
    """Processing is a background job now: the POST must return immediately."""
    from conftest import start
    body = start(pipeline.client).get_data(as_text=True)
    assert "const jobId" in body
    assert "Transcription" not in body


def test_job_status_reports_stages_then_completion(pipeline):
    import re, time
    from conftest import start
    body = start(pipeline.client).get_data(as_text=True)
    job_id = re.search(r'const jobId = "([0-9a-f]{32})"', body).group(1)
    for _ in range(200):
        state = pipeline.client.get(f"/api/jobs/{job_id}").get_json()
        if state["status"] == "done":
            break
        time.sleep(0.02)
    assert state["status"] == "done"
    assert [h["stage"] for h in state["history"]][:2] == [
        "Loading audio", "Transcribing the call"]
    assert all(h["seconds"] is not None for h in state["history"])


def test_a_malformed_job_id_is_rejected(pipeline):
    assert pipeline.client.get("/api/jobs/not-a-job").status_code == 404
    assert pipeline.client.get("/result/not-a-job").status_code == 404


def test_an_unknown_job_is_404(pipeline):
    absent = "d" * 32
    assert pipeline.client.get(f"/api/jobs/{absent}").status_code == 404
    assert pipeline.client.get(f"/result/{absent}").status_code == 404


def test_the_upload_form_posts_to_the_upload_route(pipeline):
    """A result page lives at /result/<id>, which is GET-only.

    The form had no action, so it posted to the current URL and a second
    upload from a result page returned 405 Method Not Allowed.
    """
    import re

    from conftest import upload

    body = upload(pipeline.client).get_data(as_text=True)
    action = re.search(r'<form[^>]*action="([^"]*)"', body)
    assert action, "the upload form must name the route it posts to"
    assert action.group(1) == "/"


def test_a_second_upload_from_a_result_page_works(pipeline):
    from conftest import upload

    first = upload(pipeline.client)
    assert first.status_code == 200
    second = upload(pipeline.client)
    assert second.status_code == 200


def test_the_charts_are_collapsed_by_default(pipeline):
    """They describe the audio, not the incident. The report comes first."""
    from conftest import upload

    html = upload(pipeline.client).get_data(as_text=True)
    assert '<details class="section charts"' in html
    assert 'class="section charts" id="charts" open' not in html


def test_a_run_with_no_charts_renders_no_chart_section(pipeline):
    from conftest import upload

    pipeline.plots = []
    html = upload(pipeline.client).get_data(as_text=True)
    assert 'class="section charts"' not in html


def test_every_chart_carries_a_caption(pipeline):
    """A chart nobody can interpret is decoration."""
    from conftest import upload

    html = upload(pipeline.client).get_data(as_text=True)
    assert html.count("<figcaption>") == len(pipeline.plots)


def test_charts_are_not_fetched_until_the_section_is_opened(pipeline):
    """loading="lazy" inside a closed <details> never fetches at all, not even
    on open -- the section rendered as four 2px slivers. src is set by script
    on first open instead."""
    from conftest import upload

    html = upload(pipeline.client).get_data(as_text=True)
    import re

    charts = html.split('class="section charts"', 1)[1].split("</details>", 1)[0]
    tags = re.findall(r"<img\b[^>]*>", charts)
    assert len(tags) == len(pipeline.plots)
    for tag in tags:
        assert "data-src=" in tag
        assert "src=" not in tag.replace("data-src=", "")
        assert 'loading="lazy"' not in tag
