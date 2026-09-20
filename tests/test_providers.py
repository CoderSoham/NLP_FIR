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
