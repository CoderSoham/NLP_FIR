"""LLM extraction plumbing: coercion, degradation, backend selection.

No model is loaded. The point of these tests is that the pipeline survives
every way this stage can fail, because it is an enrichment and not a
dependency.
"""
import json

import pytest

from utils import llm
from utils.extraction_schema import INCIDENT_SCHEMA, build_user_prompt
from utils.llm import LLMUnavailable, _coerce, extract_incident

MINIMAL = {"incident_type": "police", "severity": "low", "summary": "A summary."}


def test_coerce_fills_missing_keys():
    out = _coerce(MINIMAL)
    assert set(out) == set(INCIDENT_SCHEMA["properties"])
    assert out["weapons"] == [] and out["location"] is None


def test_coerce_strips_code_fences():
    out = _coerce("```json\n" + json.dumps(MINIMAL) + "\n```")
    assert out["incident_type"] == "police"


def test_coerce_ignores_prose_around_the_object():
    raw = "Sure! Here is the record:\n" + json.dumps(MINIMAL) + "\nHope that helps."
    assert _coerce(raw)["severity"] == "low"


def test_coerce_handles_nested_braces():
    payload = dict(MINIMAL, injuries=["cut to {left} hand"])
    assert _coerce(json.dumps(payload))["injuries"] == ["cut to {left} hand"]


def test_coerce_rejects_invalid_enum_values():
    assert _coerce(dict(MINIMAL, severity="catastrophic"))["severity"] == "unknown"


def test_coerce_wraps_scalars_into_arrays():
    assert _coerce(dict(MINIMAL, injuries="broken arm"))["injuries"] == ["broken arm"]


def test_weapons_without_the_required_quote_are_rejected():
    """weapons is an object array now: every entry must carry its evidence."""
    out = _coerce(dict(MINIMAL, weapons=["gun", {"item": "knife"}]))
    assert out["weapons"] == []
    assert len(out["_rejected"]["weapons"]) == 2


def test_weapons_with_a_quote_are_kept_by_coercion():
    out = _coerce(dict(MINIMAL, weapons=[{"item": "knife", "quote": "he has a knife"}]))
    assert out["weapons"] == [{"item": "knife", "quote": "he has a knife"}]


@pytest.mark.parametrize("raw", ["not json at all", "", "{unterminated"])
def test_coerce_raises_on_unusable_output(raw):
    with pytest.raises(LLMUnavailable):
        _coerce(raw)


def test_extract_returns_none_without_a_backend(monkeypatch):
    monkeypatch.setattr(llm, "get_backend", lambda: None)
    record, meta = extract_incident("There is a fire.")
    assert record is None and meta["status"] == "unavailable"


def test_extract_skips_an_empty_transcript():
    record, meta = extract_incident("   ")
    assert record is None and meta["status"] == "skipped"


def test_a_failing_backend_degrades_rather_than_raising(monkeypatch):
    class Boom:
        name = "boom"
        def extract(self, *a, **k):
            raise RuntimeError("gpu on fire")
    monkeypatch.setattr(llm, "get_backend", lambda: Boom())
    record, meta = extract_incident("There is a fire.")
    assert record is None
    assert meta["status"] == "failed" and "gpu on fire" in meta["reason"]


def test_a_working_backend_reports_itself(monkeypatch):
    class Fake:
        name = "fake"
        def describe(self):
            return "test-model"
        def extract(self, transcript, **ctx):
            return _coerce(MINIMAL)
    monkeypatch.setattr(llm, "get_backend", lambda: Fake())
    record, meta = extract_incident("There is a fire.")
    assert meta == {"status": "ok", "backend": "fake", "model": "test-model"}
    assert record["incident_type"] == "police"


def test_backend_none_disables_the_stage(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "none")
    assert llm.resolve_backend() is None


def test_unknown_backend_is_an_error(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "wat")
    with pytest.raises(LLMUnavailable):
        llm.resolve_backend()


def test_truncation_is_declared_to_the_model():
    prompt = build_user_prompt("hello", truncated=True,
                               source_seconds=590, analysed_seconds=120)
    assert "120 seconds of a 590-second recording" in prompt
    assert "uncertainties" in prompt


def test_schema_requires_an_uncertainties_field():
    """An extraction that claims total confidence in ASR output is wrong."""
    assert "uncertainties" in INCIDENT_SCHEMA["required"]


class TestGpuSlot:
    """Only one large model may be GPU-resident at a time.

    Regression: large-v3-turbo plus a 3.8B instruct model on a 6 GB card gave
    'CUDA failed with error out of memory' on the *second* request, after the
    first had succeeded.
    """

    def setup_method(self):
        from utils import gpu
        gpu.reset()
        gpu._RELEASERS.clear()

    def test_acquiring_evicts_other_components(self, monkeypatch):
        from utils import gpu
        monkeypatch.setenv("GPU_EXCLUSIVE", "1")
        freed = []
        gpu.register("asr", lambda: (freed.append("asr"), True)[1])
        gpu.register("llm", lambda: (freed.append("llm"), True)[1])
        assert gpu.acquire("llm") == ["asr"]
        assert freed == ["asr"]

    def test_reacquiring_the_same_slot_is_a_no_op(self, monkeypatch):
        from utils import gpu
        monkeypatch.setenv("GPU_EXCLUSIVE", "1")
        calls = []
        gpu.register("asr", lambda: (calls.append(1), True)[1])
        gpu.acquire("llm")
        calls.clear()
        assert gpu.acquire("llm") == []
        assert calls == []

    def test_exclusive_mode_can_be_disabled(self, monkeypatch):
        from utils import gpu
        monkeypatch.setenv("GPU_EXCLUSIVE", "0")
        gpu.register("asr", lambda: True)
        assert gpu.acquire("llm") == []

    def test_a_releaser_that_raises_does_not_break_acquisition(self, monkeypatch):
        from utils import gpu
        monkeypatch.setenv("GPU_EXCLUSIVE", "1")
        def boom():
            raise RuntimeError("driver said no")
        gpu.register("asr", boom)
        gpu.register("other", lambda: True)
        assert gpu.acquire("llm") == ["other"]

    def test_a_component_holding_nothing_is_not_reported_as_evicted(self, monkeypatch):
        from utils import gpu
        monkeypatch.setenv("GPU_EXCLUSIVE", "1")
        gpu.register("asr", lambda: False)  # on CPU, nothing to free
        assert gpu.acquire("llm") == []


def test_unavailable_reports_why(monkeypatch):
    """A bare 'unavailable' hid three different root causes in a row."""
    from utils import llm as llm_mod
    monkeypatch.setattr(llm_mod, "_BACKEND", None)
    monkeypatch.setattr(llm_mod, "_LAST_ERROR", None)
    def boom(*a, **k):
        raise LLMUnavailable("no room on device")
    monkeypatch.setattr(llm_mod, "resolve_backend", boom)
    record, meta = extract_incident("There is a fire.")
    assert record is None
    assert meta["status"] == "unavailable"
    assert meta["reason"] == "no room on device"


def test_transient_failure_is_not_cached(monkeypatch):
    """One OOM used to disable the LLM for the life of the process."""
    from utils import llm as llm_mod
    monkeypatch.setattr(llm_mod, "_BACKEND", None)
    calls = []
    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise LLMUnavailable("transient")
        class Ok:
            name = "local"
            model_id = "m"
            def extract(self, *a, **k):
                return _coerce(MINIMAL)
        return Ok()
    monkeypatch.setattr(llm_mod, "resolve_backend", flaky)
    assert llm_mod.get_backend() is None
    assert llm_mod.get_backend() is not None
    assert len(calls) == 2


def test_invalid_enum_members_inside_arrays_are_dropped():
    """Regression: services_needed accepted 'emergencies', which is not one."""
    out = _coerce(dict(MINIMAL, services_needed=["police", "emergencies", "ems"]))
    assert out["services_needed"] == ["police", "ems"]
    assert out["_rejected"]["services_needed"] == ["emergencies"]


def test_valid_enum_members_survive():
    out = _coerce(dict(MINIMAL, services_needed=["police", "fire", "ems"]))
    assert out["services_needed"] == ["police", "fire", "ems"]
    assert "_rejected" not in out


def test_free_text_arrays_are_not_filtered():
    out = _coerce(dict(MINIMAL, injuries=["broken arm", "concussion"]))
    assert out["injuries"] == ["broken arm", "concussion"]


def test_rejected_scalar_enum_is_recorded_not_just_replaced():
    out = _coerce(dict(MINIMAL, incident_type="alien invasion"))
    assert out["incident_type"] == "unknown"
    assert out["_rejected"]["incident_type"] == ["alien invasion"]


def test_meta_model_is_a_short_string(monkeypatch):
    """Regression: llm_meta['model'] carried the whole torch module repr."""
    class Fake:
        name = "local"
        model = object()          # the torch module, as LocalBackend has
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        def describe(self):
            return f"{self.model_id} (cuda)"
        def extract(self, *a, **k):
            return _coerce(MINIMAL)
    monkeypatch.setattr(llm, "get_backend", lambda: Fake())
    _record, meta = extract_incident("There is a fire.")
    assert meta["model"] == "Qwen/Qwen2.5-1.5B-Instruct (cuda)"
    assert len(meta["model"]) < 100


def test_truncation_note_survives_missing_durations():
    """Regression: formatting a None duration raised TypeError mid-extraction."""
    prompt = build_user_prompt("hello", truncated=True)
    assert "only the beginning of a longer" in prompt
    assert "uncertainties" in prompt


def test_truncation_note_uses_durations_when_present():
    prompt = build_user_prompt("hello", truncated=True,
                               source_seconds=590.1, analysed_seconds=120.0)
    assert "first 120 seconds of a 590-second recording" in prompt


@pytest.mark.parametrize("said,expected", [
    ("ambulance", "ems"), ("EMS", "ems"), ("emt", "ems"), ("paramedics", "ems"),
    ("police", "police"), ("law enforcement", "police"), ("Sheriff", "police"),
    ("fire department", "fire"), ("firefighters", "fire"),
])
def test_service_aliases_are_normalised(said, expected):
    """Models name services the way a dispatcher would, not the way the enum does."""
    assert _coerce(dict(MINIMAL, services_needed=[said]))["services_needed"] == [expected]


def test_a_non_service_is_still_rejected():
    """'emergencies' is not a service -- it means the model did not decide."""
    out = _coerce(dict(MINIMAL, services_needed=["emergencies", "police"]))
    assert out["services_needed"] == ["police"]
    assert out["_rejected"]["services_needed"] == ["emergencies"]


def test_aliases_are_deduplicated():
    out = _coerce(dict(MINIMAL, services_needed=["ambulance", "EMS", "emt"]))
    assert out["services_needed"] == ["ems"]


def test_a_structured_location_is_flattened_not_crashed():
    """Regression: a model returned {"street":..,"city":..} and killed the harness."""
    out = _coerce(dict(MINIMAL, location={"street": "714 Court", "city": "Flint"}))
    assert out["location"] == "714 Court, Flint"
    assert out["_rejected"]["location"]


def test_a_list_location_is_flattened():
    out = _coerce(dict(MINIMAL, location=["24th Street", "near Mueller Brass"]))
    assert out["location"] == "24th Street, near Mueller Brass"


def test_a_numeric_scalar_becomes_a_string():
    assert _coerce(dict(MINIMAL, callback_number=5552443))["callback_number"] == "5552443"


def test_an_empty_dict_location_becomes_none():
    assert _coerce(dict(MINIMAL, location={}))["location"] is None


# ---- Failover between hosted providers --------------------------------------
# A free tier returning 429 for an hour is the expected case, not the unlucky
# one. If a second key is already in the environment, the stage should use it.

class _Fallible:
    """A hosted backend that fails a fixed number of times, then succeeds."""

    is_hosted = True

    def __init__(self, name, failures=0):
        self.name = name
        self.failures = failures
        self.calls = 0

    def describe(self):
        return f"model @ {self.name}"

    def extract(self, *_a, **_k):
        self.calls += 1
        if self.calls <= self.failures:
            raise LLMUnavailable("429 rate limit exceeded")
        return _coerce(MINIMAL)


def test_fallback_chain_lists_only_providers_with_keys(monkeypatch):
    monkeypatch.delenv("LLM_FALLBACK_BACKENDS", raising=False)
    for cfg in llm.llm_providers.PROVIDERS.values():
        if cfg.get("key_env"):
            monkeypatch.delenv(cfg["key_env"], raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert llm.fallback_backends("nvidia") == ["groq"]


def test_fallback_chain_excludes_the_primary(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert "nvidia" not in llm.fallback_backends("nvidia")


def test_fallback_chain_can_be_disabled(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_BACKENDS", "")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert llm.fallback_backends("nvidia") == []


def test_fallback_chain_is_explicit_when_set(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_BACKENDS", "groq, cerebras")
    assert llm.fallback_backends("nvidia") == ["groq", "cerebras"]


def test_a_rate_limited_primary_falls_over_to_the_next_provider(monkeypatch):
    primary, alt = _Fallible("nvidia", failures=1), _Fallible("groq")
    monkeypatch.setattr(llm, "get_backend", lambda: primary)
    monkeypatch.setattr(llm, "fallback_backends", lambda _p: ["groq"])
    monkeypatch.setattr(llm, "OpenAICompatibleBackend", lambda _n: alt)

    record, meta = extract_incident("There is a fire.")

    assert record is not None
    assert meta["status"] == "ok"
    assert meta["backend"] == "groq"
    assert meta["failed_over_from"] == "nvidia"
    assert [a["backend"] for a in meta["attempts"]] == ["nvidia", "groq"]


def test_a_working_primary_is_not_retried_elsewhere(monkeypatch):
    primary, alt = _Fallible("nvidia"), _Fallible("groq")
    monkeypatch.setattr(llm, "get_backend", lambda: primary)
    monkeypatch.setattr(llm, "fallback_backends", lambda _p: ["groq"])
    monkeypatch.setattr(llm, "OpenAICompatibleBackend", lambda _n: alt)

    _record, meta = extract_incident("There is a fire.")

    assert alt.calls == 0
    assert "attempts" not in meta       # a clean run stays uncluttered


def test_every_provider_failing_reports_the_primary_reason(monkeypatch):
    primary, alt = _Fallible("nvidia", failures=9), _Fallible("groq", failures=9)
    monkeypatch.setattr(llm, "get_backend", lambda: primary)
    monkeypatch.setattr(llm, "fallback_backends", lambda _p: ["groq"])
    monkeypatch.setattr(llm, "OpenAICompatibleBackend", lambda _n: alt)

    record, meta = extract_incident("There is a fire.")

    assert record is None
    assert meta["status"] == "failed"
    assert meta["backend"] == "nvidia"          # blame the configured one
    assert "429" in meta["reason"]
    assert len(meta["attempts"]) == 2


def test_a_local_backend_does_not_fall_over(monkeypatch):
    """A local failure is this machine's; a second local model fails the same."""
    local = _Fallible("local", failures=9)
    local.is_hosted = False
    alt = _Fallible("groq")
    monkeypatch.setattr(llm, "get_backend", lambda: local)
    monkeypatch.setattr(llm, "fallback_backends", lambda _p: ["groq"])
    monkeypatch.setattr(llm, "OpenAICompatibleBackend", lambda _n: alt)

    record, meta = extract_incident("There is a fire.")

    assert record is None and alt.calls == 0
    assert meta["backend"] == "local"


def test_local_backend_is_not_hosted():
    """The pipeline overlaps the LLM with GPU stages only when it is hosted."""
    assert llm.LocalBackend.is_hosted is False
    assert llm.AnthropicBackend.is_hosted is True


# ---- CPU headroom guard -----------------------------------------------------
# An out-of-memory condition on CPU does not raise, it swaps. A 3B model at
# float32 ran for nearly nine minutes producing nothing, and the only symptom
# was laptop fan noise. See ISSUE-037.

@pytest.mark.parametrize("model_id,expected", [
    ("Qwen/Qwen2.5-1.5B-Instruct", 1.5),
    ("Qwen/Qwen2.5-3B-Instruct", 3.0),
    ("meta-llama/Llama-3.1-8B-Instruct", 8.0),
    ("nvidia/nemotron-3-ultra-550b-a55b", 550.0),
    ("google/flan-t5-base", None),
    ("some/model-without-a-size", None),
])
def test_parameter_count_is_read_from_the_name(model_id, expected):
    assert llm._params_from_name(model_id) == expected


def test_a_model_that_fits_is_allowed(monkeypatch):
    monkeypatch.setattr(llm.os, "sysconf",
                        lambda n: 10 ** 9 if n == "SC_AVPHYS_PAGES" else 16)
    llm.check_cpu_headroom("Qwen/Qwen2.5-1.5B-Instruct")     # must not raise


def test_a_model_that_would_swap_is_refused(monkeypatch):
    # 2 GB free against a 7B model needing ~19 GB.
    monkeypatch.setattr(llm.os, "sysconf",
                        lambda n: 2 * 10 ** 9 if n == "SC_AVPHYS_PAGES" else 1)
    with pytest.raises(LLMUnavailable, match="would swap"):
        llm.check_cpu_headroom("Qwen/Qwen2.5-7B-Instruct")


def test_an_unsized_model_is_not_blocked(monkeypatch):
    """Better to attempt a load than to refuse on a name we cannot parse."""
    monkeypatch.setattr(llm.os, "sysconf", lambda n: 1)
    llm.check_cpu_headroom("google/flan-t5-base")            # must not raise


def test_a_platform_without_sysconf_is_not_blocked(monkeypatch):
    def unsupported(_name):
        raise ValueError("unrecognised configuration name")

    monkeypatch.setattr(llm.os, "sysconf", unsupported)
    llm.check_cpu_headroom("Qwen/Qwen2.5-7B-Instruct")       # must not raise


def test_the_cpu_path_does_not_load_at_float32():
    """Regression: float32 put a 3B model at 12.4 GB on a 16 GB machine.

    Reads the code because the alternative is loading a model. The fix was
    designed, written into the vault, and then never applied -- it ran for ten
    more days and was found by the fans getting loud.
    """
    import inspect

    source = inspect.getsource(llm.LocalBackend._ensure_loaded)
    code = [ln for ln in source.splitlines()
            if not ln.lstrip().startswith("#")]
    assert any("bfloat16" in ln for ln in code)
    assert not any("float32" in ln for ln in code)
