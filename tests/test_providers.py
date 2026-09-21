"""OpenAI-compatible provider plumbing. No network."""
import json
import urllib.error
import io

import pytest

from utils import llm, llm_providers
from utils.llm import LLMUnavailable


def test_every_provider_has_the_fields_the_backend_needs():
    for name, cfg in llm_providers.PROVIDERS.items():
        assert cfg["base_url"].startswith("http"), name
        assert cfg["default_model"], name
        assert "hosted" in cfg and "note" in cfg, name


def test_local_providers_need_no_key():
    for name in ("ollama", "llamacpp"):
        assert llm_providers.PROVIDERS[name]["key_env"] is None
        assert llm_providers.PROVIDERS[name]["hosted"] is False


def test_hosted_providers_all_declare_a_key_variable():
    for name, cfg in llm_providers.PROVIDERS.items():
        if cfg["hosted"]:
            assert cfg["key_env"], name


def test_resolve_applies_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "some/other-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    cfg = llm_providers.resolve("nvidia")
    assert cfg["model"] == "some/other-model"
    assert cfg["base_url"] == "http://127.0.0.1:9999/v1"
    assert cfg["api_key"] == "nvapi-test"


def test_a_hosted_provider_without_its_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_BACKEND", "groq")
    with pytest.raises(LLMUnavailable, match="GROQ_API_KEY"):
        llm.resolve_backend()


def test_a_local_provider_needs_no_key(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    backend = llm.resolve_backend()
    assert backend.name == "ollama"
    assert "ollama" in backend.describe()


def test_unknown_backend_lists_the_valid_ones(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "wat")
    with pytest.raises(LLMUnavailable, match="nvidia"):
        llm.resolve_backend()


def _fake_http_error(code, body):
    return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(body.encode()))


def test_json_mode_is_retried_without_response_format(monkeypatch):
    """Not every provider accepts response_format; some 400 rather than ignore."""
    calls = []

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()

    def fake_urlopen(req, timeout=None):
        payload = json.loads(req.data)
        calls.append(payload)
        if "response_format" in payload:
            raise _fake_http_error(400, "unsupported parameter: response_format")
        return Resp()

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    cfg = llm_providers.resolve("ollama")
    assert llm_providers.chat_completion(cfg, [{"role": "user", "content": "x"}]) == "{}"
    assert len(calls) == 2 and "response_format" not in calls[1]


def test_an_unreachable_provider_names_itself(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("Connection refused")
    # Without stubbing sleep this test really waits out the backoff -- it was
    # 30 of the suite's 35 seconds on its own.
    monkeypatch.setattr(llm_providers.time, "sleep", lambda _s: None)
    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", boom)
    cfg = llm_providers.resolve("ollama")
    with pytest.raises(RuntimeError, match="cannot reach ollama"):
        llm_providers.chat_completion(cfg, [{"role": "user", "content": "x"}])


def test_eval_harness_loads_project_config():
    """Regression: eval/run.py resolved a backend without importing config,
    so `.env` was never read and a configured key looked missing."""
    source = open("eval/run.py").read()
    assert "import config" in source
    assert source.index("import config") < source.index("def score")


# ---- the stage budget ------------------------------------------------------
# Since the local classifiers stopped running whenever the extraction
# succeeds, this call *is* the request. Five retries at a 300s timeout is
# twenty-five minutes behind a spinner.

class _Ok:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self):
        return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()


def test_groqs_json_validate_failure_retries_in_plain_text(monkeypatch):
    """Groq accepts response_format and then 400s when the output does not
    satisfy it, with an empty failed_generation. Same problem as a provider
    rejecting the field, same remedy."""
    calls = []

    def fake_urlopen(req, timeout=None):
        payload = json.loads(req.data)
        calls.append(payload)
        if "response_format" in payload:
            raise _fake_http_error(400, json.dumps({"error": {
                "message": "Failed to validate JSON.",
                "code": "json_validate_failed", "failed_generation": ""}}))
        return _Ok()

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    cfg = llm_providers.resolve("ollama")
    assert llm_providers.chat_completion(cfg, [{"role": "user", "content": "x"}]) == "{}"
    assert len(calls) == 2 and "response_format" not in calls[1]


def test_retries_stop_when_the_budget_is_spent(monkeypatch):
    attempts = []
    clock = [1000.0]

    def fake_urlopen(req, timeout=None):
        attempts.append(timeout)
        clock[0] += 20                      # each attempt burns 20s
        raise _fake_http_error(429, "rate limited")

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_providers.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(llm_providers.time, "sleep",
                        lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setenv("LLM_DEADLINE", "50")

    cfg = llm_providers.resolve("ollama")
    with pytest.raises(RuntimeError, match="did not answer within the 50s budget"):
        llm_providers.chat_completion(cfg, [{"role": "user", "content": "x"}])

    # Stopped on the clock, not on the retry count -- LLM_RETRIES is 4.
    assert len(attempts) < 4


def test_each_attempt_is_capped_by_what_the_budget_has_left(monkeypatch):
    """A 300s socket timeout inside a 180s budget is a 300s socket timeout."""
    seen = []
    clock = [1000.0]

    def fake_urlopen(req, timeout=None):
        seen.append(timeout)
        return _Ok()

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_providers.time, "monotonic", lambda: clock[0])
    monkeypatch.setenv("LLM_DEADLINE", "45")
    monkeypatch.setenv("LLM_TIMEOUT", "300")

    cfg = llm_providers.resolve("ollama")
    llm_providers.chat_completion(cfg, [{"role": "user", "content": "x"}])
    assert seen == [45]


def test_backoff_never_sleeps_past_the_deadline(monkeypatch):
    """Sleeping past the budget spends the remainder waiting rather than on
    one more try, and returns the failure later than it was known."""
    clock = [1000.0]
    slept = []
    monkeypatch.setattr(llm_providers.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(llm_providers.time, "sleep", slept.append)

    llm_providers._backoff(attempt=5, deadline=clock[0] + 3)
    assert slept == [3]

    slept.clear()
    llm_providers._backoff(attempt=5, deadline=clock[0] - 10)
    assert slept == [0.0]


def test_a_generous_budget_still_allows_the_configured_retries(monkeypatch):
    """The deadline is a ceiling, not a replacement for the retry count."""
    attempts = []
    clock = [1000.0]

    def fake_urlopen(req, timeout=None):
        attempts.append(timeout)
        clock[0] += 1
        raise _fake_http_error(503, "busy")

    monkeypatch.setattr(llm_providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_providers.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(llm_providers.time, "sleep",
                        lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setenv("LLM_DEADLINE", "3600")
    monkeypatch.setenv("LLM_RETRIES", "2")

    cfg = llm_providers.resolve("ollama")
    with pytest.raises(RuntimeError, match="HTTP 503"):
        llm_providers.chat_completion(cfg, [{"role": "user", "content": "x"}])
    assert len(attempts) == 3            # the first, plus two retries


def test_the_eval_model_flag_targets_the_right_backend():
    """`--model X --backend groq` set LOCAL_LLM_MODEL, which only the local
    backend reads -- so it ran Groq's default model and printed its name. A
    bake-off between two hosted models would have compared one with itself."""
    from eval.run import model_env

    assert model_env("local") == "LOCAL_LLM_MODEL"
    for hosted in ("groq", "nvidia", "openrouter", "cerebras", "anthropic"):
        assert model_env(hosted) == "LLM_MODEL"


# ---- a model id belongs to one provider ------------------------------------

def test_the_global_model_override_applies_to_the_chosen_backend(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b")
    assert llm_providers.resolve("groq")["model"] == "openai/gpt-oss-120b"


def test_the_global_override_does_not_follow_a_failover(monkeypatch):
    """Observed live: a 429 from Groq fell over to NVIDIA carrying Groq's
    model id, so the fallback that exists to keep the request alive asked for
    a model NVIDIA does not serve."""
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b")
    fallback = llm_providers.resolve("nvidia", primary=False)
    assert fallback["model"] == llm_providers.PROVIDERS["nvidia"]["default_model"]


def test_the_global_base_url_does_not_follow_a_failover_either(monkeypatch):
    """Worse than the model: it would send NVIDIA's key to Groq's endpoint."""
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    assert llm_providers.resolve("nvidia", primary=False)["base_url"] == \
        llm_providers.PROVIDERS["nvidia"]["base_url"]
    assert llm_providers.resolve("nvidia")["base_url"] == "http://localhost:11434/v1"


def test_a_provider_specific_override_is_honoured_even_on_a_failover(monkeypatch):
    """Naming the provider in the variable is saying which one you meant."""
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("LLM_MODEL_NVIDIA", "nvidia/nemotron-3-super-120b-a12b")
    assert llm_providers.resolve("nvidia", primary=False)["model"] == \
        "nvidia/nemotron-3-super-120b-a12b"


def test_a_provider_specific_override_beats_the_global_one(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "global-model")
    monkeypatch.setenv("LLM_MODEL_GROQ", "specific-model")
    assert llm_providers.resolve("groq")["model"] == "specific-model"


def test_the_fallback_backend_is_constructed_as_a_fallback(monkeypatch):
    """The end-to-end version: the wiring, not just resolve()."""
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("NVIDIA_API_KEY", "key")
    backend = llm.OpenAICompatibleBackend("nvidia", primary=False)
    assert "gpt-oss" not in backend.describe()
