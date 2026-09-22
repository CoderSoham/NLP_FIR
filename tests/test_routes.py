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
    """Wherever the form is rendered, it must name the route it posts to.

    A result page lives at /result/<id>, which is GET-only. The form had no
    action, so it posted to the current URL and a second upload from a result
    page returned 405 Method Not Allowed.
    """
    import re

    body = pipeline.client.get("/").get_data(as_text=True)
    action = re.search(r'<form[^>]*action="([^"]*)"', body)
    assert action, "the upload form must name the route it posts to"
    assert action.group(1) == "/"


def test_a_result_page_offers_a_way_back_to_the_upload_route(pipeline):
    """The report no longer carries a drop zone -- a full upload form on a
    finished report is clutter -- so the escape hatch has to be somewhere."""
    from conftest import upload

    body = upload(pipeline.client).get_data(as_text=True)
    assert 'href="/"' in body


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


def test_the_page_says_which_path_classified_the_call(pipeline):
    """Without the local classifiers there is no independent second opinion,
    and the disagreement banner cannot fire on type or severity. A reader is
    entitled to know which of those two situations they are looking at."""
    from conftest import upload

    pipeline.llm_meta = {"status": "ok", "model": "test", "second_opinion": False}
    html = upload(pipeline.client).get_data(as_text=True)
    assert "The extraction model alone" in html
    assert "LOCAL_ANALYSIS=always" in html


def test_a_second_opinion_is_reported_when_one_was_taken(pipeline):
    from conftest import upload

    pipeline.llm_meta = {"status": "ok", "model": "test", "second_opinion": True}
    html = upload(pipeline.client).get_data(as_text=True)
    assert "ran as a second opinion" in html


def test_the_result_page_can_play_the_recording(pipeline):
    """app.py builds audio_url from data['audio_filename']. The pipeline never
    set it, so the check was always false and the player only ever appeared on
    the progress page -- not on the report, where you would check a passage
    against the audio."""
    from conftest import upload

    html = upload(pipeline.client).get_data(as_text=True)
    assert "<audio" in html
    assert "/audio/" in html


def test_the_report_renders_with_only_the_fields_that_always_exist(pipeline):
    """Every stage is allowed to fail; the report is not.

    A missing optional field used to take the whole page down with a 500 --
    `analysed_duration_s` did exactly that the moment the audio player started
    rendering. The transcript is what makes it a report; everything else is a
    section that should simply not appear.
    """
    import app as app_module
    from flask import render_template

    minimal = {
        "report_id": "a" * 32,
        "transcription": "there is a fire on alameda",
        "timestamp": "2026-01-01 00:00:00",
        "emergency_type": "fire",
        "severity": "high",
        "emergency_response": {"suggestions": []},
        "entities": [],
        "plots": [],
    }
    with app_module.app.test_request_context("/"):
        html = render_template("index.html", **minimal)
    assert "Incident report" in html
    assert "there is a fire on alameda" in html


def test_the_upload_page_renders_with_no_context_at_all(pipeline):
    """GET / passes nothing but the template name."""
    body = pipeline.client.get("/").get_data(as_text=True)
    assert body.count("dropzone") > 1
    assert "Analyse recording" in body


def test_a_raised_severity_names_the_words_that_raised_it(pipeline):
    """Keyword matching cannot tell a threat from a figure of speech, so the
    page shows the evidence rather than only the verdict."""
    import html as html_lib
    import re

    import app as app_module
    from flask import render_template

    ctx = dict(report_id="a" * 32, transcription="x", timestamp="t",
               emergency_type="police", severity="critical",
               severity_before_floor="medium", severity_source="keyword floor",
               severity_floor="critical", severity_floor_words=["shot", "shots"],
               emergency_response={"suggestions": []}, entities=[], plots=[])
    with app_module.app.test_request_context("/"):
        page = render_template("index.html", **ctx)

    note = re.search(r'<span class="triage-note">(.*?)</span>', page, re.S)
    assert note, "a raised severity must say it was raised"
    text = html_lib.unescape(" ".join(note.group(1).split()))
    assert "raised from Medium" in text
    assert "shot, shots" in text
    # Regression: joining on '", "' escaped to &#34; and rendered as a mix of
    # curly and straight quotes.
    assert '"' not in text


def test_a_severity_the_model_already_got_right_is_not_annotated(pipeline):
    import app as app_module
    from flask import render_template

    ctx = dict(report_id="a" * 32, transcription="x", timestamp="t",
               emergency_type="police", severity="critical",
               severity_before_floor="critical", severity_source="model",
               emergency_response={"suggestions": []}, entities=[], plots=[])
    with app_module.app.test_request_context("/"):
        page = render_template("index.html", **ctx)
    assert "triage-note" not in page
