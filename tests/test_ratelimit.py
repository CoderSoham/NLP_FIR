"""Rate limiter and API-key auth. Pure logic plus the wired routes."""
import io

import pytest

from utils.ratelimit import RateLimiter, api_key_ok


class FakeRequest:
    def __init__(self, headers=None, remote_addr="1.2.3.4"):
        self.headers = headers or {}
        self.remote_addr = remote_addr


def test_allows_up_to_the_limit():
    rl = RateLimiter(3, 60)
    assert [rl.check("a", now=100)[0] for _ in range(3)] == [True, True, True]


def test_blocks_past_the_limit():
    rl = RateLimiter(2, 60)
    rl.check("a", now=100)
    rl.check("a", now=100)
    allowed, retry_after = rl.check("a", now=100)
    assert allowed is False
    assert retry_after > 0


def test_window_slides():
    rl = RateLimiter(1, 60)
    assert rl.check("a", now=100)[0] is True
    assert rl.check("a", now=130)[0] is False
    assert rl.check("a", now=161)[0] is True


def test_keys_are_independent():
    rl = RateLimiter(1, 60)
    assert rl.check("a", now=100)[0] is True
    assert rl.check("b", now=100)[0] is True


def test_zero_limit_disables():
    assert RateLimiter(0, 60).check("a", now=100)[0] is True


def test_forget_bounds_the_map():
    rl = RateLimiter(5, 60)
    rl.check("a", now=100)
    rl.forget(older_than_seconds=60, now=1000)
    assert rl._hits == {}


def test_unset_api_key_means_open():
    assert api_key_ok(FakeRequest(), "") is True


def test_api_key_must_match():
    assert api_key_ok(FakeRequest({"X-API-Key": "right"}), "right") is True
    assert api_key_ok(FakeRequest({"X-API-Key": "wrong"}), "right") is False
    assert api_key_ok(FakeRequest(), "right") is False


def test_forwarded_header_ignored_unless_trusted():
    from utils.ratelimit import client_key
    req = FakeRequest({"X-Forwarded-For": "9.9.9.9"}, remote_addr="1.2.3.4")
    assert client_key(req, trust_proxy=False) == "1.2.3.4"
    assert client_key(req, trust_proxy=True) == "9.9.9.9"


def upload(client):
    return client.post(
        "/", data={"audio_file": (io.BytesIO(b"x"), "call.mp3")},
        content_type="multipart/form-data")


def test_route_returns_429_with_retry_after(pipeline, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "limiter", RateLimiter(2, 600))
    assert upload(pipeline.client).status_code == 200
    assert upload(pipeline.client).status_code == 200
    r = upload(pipeline.client)
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) > 0


def test_route_requires_api_key_when_set(pipeline, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, "API_KEY", "s3cret")
    assert upload(pipeline.client).status_code == 401
    r = pipeline.client.post(
        "/", data={"audio_file": (io.BytesIO(b"x"), "call.mp3")},
        content_type="multipart/form-data", headers={"X-API-Key": "s3cret"})
    assert r.status_code == 200


def test_fetching_an_artefact_is_not_rate_limited(pipeline, monkeypatch):
    """Only pipeline runs are limited; the result page loads four images."""
    import app as app_module
    monkeypatch.setattr(app_module, "limiter", RateLimiter(1, 600))
    upload(pipeline.client)
    for _ in range(5):
        assert pipeline.client.get(
            f"/plot/{pipeline.report_id}/waveform").status_code == 200
