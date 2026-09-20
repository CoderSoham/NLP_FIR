"""Structured incident extraction, with pluggable backends.

Three backends behind one interface:

- **local** (default) - an instruct model on your own hardware via transformers.
  Nothing leaves the machine, which for emergency-call transcripts containing
  names, addresses and medical detail is a property worth keeping by default.
- **anthropic** - Claude, when the operator sets an API key and accepts that
  transcripts are sent to a third party.
- **none** - the stage is skipped and the rest of the pipeline runs unchanged.

The LLM is an *enrichment* stage. Every consumer must work when it returns
None, because it will: no key, no model downloaded, out of memory, malformed
output. The classical NLP path stays the floor, not the fallback.
"""
import json
import os
import re

from utils import gpu
from utils import llm_providers
from utils.evidence import verify_evidence
from utils.extraction_schema import (INCIDENT_SCHEMA, SYSTEM_PROMPT,
                                     build_user_prompt, normalise_service)

_BACKEND = None
_LAST_ERROR = None


class LLMUnavailable(Exception):
    """The stage cannot run. Callers degrade; they do not fail."""


def _coerce(raw):
    """Validate and normalise a model's JSON against the schema.

    Local models return prose around their JSON, wrap it in fences, or emit a
    trailing comma. Rather than demand perfection from a 3B model, extract the
    first balanced object and fill missing keys with schema-appropriate empties.
    """
    if isinstance(raw, dict):
        data = raw
    else:
        text = (raw or "").strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
        start = text.find("{")
        if start == -1:
            raise LLMUnavailable("no JSON object in model output")
        depth, end = 0, None
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            raise LLMUnavailable("unterminated JSON object in model output")
        try:
            data = json.loads(text[start:end])
        except ValueError as exc:
            raise LLMUnavailable(f"invalid JSON from model: {exc}") from exc

    out, rejected = {}, {}
    for key, spec in INCIDENT_SCHEMA["properties"].items():
        value = data.get(key)
        types = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
        if value is None:
            out[key] = [] if "array" in types else ("" if key == "summary" else None)
            continue
        if "array" in types and not isinstance(value, list):
            value = [value]
        if "array" in types:
            item_spec = spec.get("items") or {}
            if item_spec.get("type") == "object":
                # Structured array members are kept as dicts; only their
                # required keys are enforced.
                required = item_spec.get("required", [])
                kept = []
                for v in value:
                    if isinstance(v, dict) and all(k in v for k in required):
                        kept.append({k: str(v[k]) for k in item_spec["properties"]
                                     if k in v})
                    else:
                        rejected.setdefault(key, []).append(v)
                value = kept
            else:
                value = [str(v) for v in value if v is not None]
            # Members of a constrained array must be checked too. Only the
            # scalar `enum` was validated before, so services_needed accepted
            # "emergencies" -- not a service -- and it reached the report.
            item_enum = (spec.get("items") or {}).get("enum")
            if item_enum and value and isinstance(value[0], dict):
                item_enum = None
            if key == "services_needed":
                # Normalise before validating: a model that says "ambulance"
                # identified the right service and spelled it differently.
                normalised = []
                for v in value:
                    mapped = normalise_service(v)
                    if mapped:
                        normalised.append(mapped)
                    else:
                        rejected.setdefault(key, []).append(v)
                seen = set()
                value = [v for v in normalised if not (v in seen or seen.add(v))]
                item_enum = None
            if item_enum:
                dropped = [v for v in value if v not in item_enum]
                value = [v for v in value if v in item_enum]
                if dropped:
                    rejected.setdefault(key, []).extend(dropped)
        # Scalar string fields must be strings. A model returning
        # {"street": "...", "city": "..."} for `location` crashed the scorer,
        # and would reach the report as a dict. Flatten what can be flattened,
        # reject what cannot, and record either way.
        if "array" not in types and "string" in types and value is not None:
            if isinstance(value, dict):
                parts = [str(v) for v in value.values() if v not in (None, "")]
                rejected.setdefault(key, []).append(value)
                value = ", ".join(parts) if parts else None
            elif isinstance(value, list):
                rejected.setdefault(key, []).append(value)
                value = ", ".join(str(v) for v in value if v is not None) or None
            elif not isinstance(value, str):
                value = str(value)

        if spec.get("enum") and value not in spec["enum"]:
            rejected.setdefault(key, []).append(value)
            value = "unknown" if "unknown" in spec["enum"] else None
        out[key] = value

    # Surfaced rather than dropped silently: a model repeatedly emitting
    # invalid members is information about the prompt, not noise.
    if rejected:
        out["_rejected"] = rejected
    return out


class AnthropicBackend:
    """Claude via the official SDK, using structured outputs."""

    name = "anthropic"
    # Network-bound, so it can run alongside the GPU stages. See
    # `utils.audio_utils` for why that distinction is load-bearing.
    is_hosted = True

    def describe(self):
        return self.model

    def __init__(self, model=None, api_key=None):
        try:
            import anthropic
        except ImportError as exc:
            raise LLMUnavailable(
                "anthropic package not installed; pip install anthropic") from exc
        self._anthropic = anthropic
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()

    def extract(self, transcript, **context):
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": "high",
                    "format": {
                        "type": "json_schema",
                        "schema": INCIDENT_SCHEMA,
                    },
                },
                messages=[{"role": "user",
                           "content": build_user_prompt(transcript, **context)}],
            )
        except self._anthropic.APIError as exc:
            raise LLMUnavailable(f"Anthropic API error: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMUnavailable("model declined this transcript")

        text = "".join(b.text for b in response.content if b.type == "text")
        return _coerce(text)


class LocalBackend:
    """An instruct model on this machine, via transformers.

    Loads lazily and *reloads* after eviction. The GPU slot manager may unload
    this model to make room for ASR at any point between requests, so a backend
    that could only be used once was worse than no backend at all.
    """

    name = "local"
    # Shares the GPU with ASR, NER and the classifiers, so it must never be
    # run concurrently with them.
    is_hosted = False

    def describe(self):
        return f"{self.model_id} ({self.device})"

    def __init__(self, model_id=None, device=None):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LLMUnavailable("transformers/torch not available") from exc

        from utils.llm_hardware import recommend
        self._AutoModel = AutoModelForCausalLM
        self._AutoTokenizer = AutoTokenizer
        self._torch = torch
        self.model_id = model_id or os.environ.get("LOCAL_LLM_MODEL") or recommend()[0]
        self.device = (device or os.environ.get("LOCAL_LLM_DEVICE")
                       or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.tokenizer = None
        # Bounded so one long transcript cannot enlarge the KV cache until it
        # no longer fits beside the model.
        self.max_input_tokens = int(os.environ.get("LOCAL_LLM_MAX_INPUT", 3072))
        self.max_new_tokens = int(os.environ.get("LOCAL_LLM_MAX_NEW", 768))
        gpu.register("llm", self._unload)
        self._ensure_loaded()

    def _ensure_loaded(self):
        if self.model is not None:
            return
        if self.device == "cuda":
            gpu.acquire("llm")
        try:
            if self.tokenizer is None:
                self.tokenizer = self._AutoTokenizer.from_pretrained(
                    self.model_id, trust_remote_code=False)
            self.model = self._AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=(self._torch.float16 if self.device == "cuda"
                             else self._torch.float32),
                device_map=self.device,
                trust_remote_code=False,
            )
            self.model.eval()
        except Exception as exc:
            self.model = None
            gpu.empty_cache()
            if "out of memory" in str(exc).lower() and self.device == "cuda":
                # Evict whatever else is resident and try once more before
                # declaring the stage unavailable for this request.
                gpu.acquire("llm")
                gpu.empty_cache()
                try:
                    self.model = self._AutoModel.from_pretrained(
                        self.model_id, dtype=self._torch.float16,
                        device_map="cuda", trust_remote_code=False)
                    self.model.eval()
                    return
                except Exception:
                    self.model = None
                    gpu.empty_cache()
            # Keep the message short: a transformers error can carry the whole
            # model repr, which is thousands of lines in a log line.
            raise LLMUnavailable(
                f"could not load {self.model_id}: {type(exc).__name__}: "
                f"{str(exc).splitlines()[0][:200]}") from exc

    def _unload(self):
        """Actually free the GPU memory.

        Setting the attribute to None is not enough. `device_map` installs
        accelerate hooks that hold references, and CPython will not collect the
        parameter storages while those live -- an earlier version did exactly
        that and freed nothing, so the next ASR load hit OOM with the LLM still
        resident. Moving to meta/CPU and dropping the modules first is what
        makes the storages collectable.
        """
        if self.model is None or self.device != "cuda":
            return False
        try:
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(self.model, recurse=True)
        except Exception:
            pass
        try:
            self.model.to("cpu")
        except Exception:
            pass
        self.model = None
        gpu.empty_cache()
        return True

    def extract(self, transcript, **context):
        self._ensure_loaded()
        # The full JSON Schema, not a shortened spec. Measured on the eval
        # set: full schema 11/18, a hand-written 360-token prose spec 7/18, an
        # auto-generated 123-token field list 6/18. The schema's structure --
        # not merely its descriptions -- is doing work, presumably by priming
        # the model to emit JSON of the same shape. Prompt length is bought
        # back in memory, not in tokens; see LOCAL_LLM_MAX_INPUT and the
        # prefill note in utils/llm_hardware.py.
        schema = json.dumps(INCIDENT_SCHEMA["properties"], indent=2)
        user = (build_user_prompt(transcript, **context)
                + "\n\nReturn a single JSON object with exactly these keys:\n"
                + schema + "\n\nRespond with JSON only, no commentary.")

        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user}]
        inputs = generated = None
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=self.max_input_tokens).to(self.device)
            with self._torch.inference_mode():
                # Greedy: extraction should be reproducible for the same
                # transcript. top_k/top_p are cleared because several chat
                # models ship sampling defaults in generation_config that warn
                # (and would otherwise be ignored) under do_sample=False.
                generated = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens,
                    do_sample=False, temperature=None, top_k=None, top_p=None,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id)
            text = self.tokenizer.decode(
                generated[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)
        except Exception as exc:
            raise LLMUnavailable(
                f"local generation failed: {type(exc).__name__}: "
                f"{str(exc).splitlines()[0][:200]}") from exc
        finally:
            # The KV cache and the input/output tensors are the largest
            # transient allocations in the process. Without dropping them and
            # returning the blocks to the driver, a second extraction in the
            # same process runs out of memory even though the first fitted --
            # observed on the first eval run, sample 1 passing and 2 and 3
            # failing with OutOfMemoryError.
            del inputs, generated
            if self.device == "cuda":
                gpu.empty_cache()
        return _coerce(text)


class OpenAICompatibleBackend:
    """Any provider speaking the OpenAI /chat/completions shape.

    One implementation reaches NVIDIA's free tier, Groq, OpenRouter, Cerebras,
    Mistral, and a local Ollama or llama.cpp server. The last of those is how a
    quantised model runs on a GPU that `bitsandbytes` will not support.
    """

    def __init__(self, provider):
        self.cfg = llm_providers.resolve(provider)
        self.name = provider
        # ollama and llama.cpp speak the same protocol but hold their own VRAM
        # on this machine, so they are not "hosted" for scheduling purposes.
        self.is_hosted = bool(self.cfg.get("hosted"))
        if self.cfg["key_env"] and not self.cfg["api_key"]:
            raise LLMUnavailable(
                f"{provider} needs {self.cfg['key_env']} in the environment")

    def describe(self):
        return f"{self.cfg['model']} @ {self.name}"

    def extract(self, transcript, **context):
        schema = json.dumps(INCIDENT_SCHEMA["properties"], indent=2)
        user = (build_user_prompt(transcript, **context)
                + "\n\nReturn a single JSON object with exactly these keys:\n"
                + schema + "\n\nRespond with JSON only, no commentary.")
        try:
            text = llm_providers.chat_completion(
                self.cfg,
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user}])
        except Exception as exc:
            raise LLMUnavailable(str(exc)[:300]) from exc
        return _coerce(text)


def resolve_backend(name=None):
    """Pick a backend. `auto` prefers a key if present, else local, else none."""
    name = (name or os.environ.get("LLM_BACKEND", "auto")).lower()
    if name == "none":
        return None
    if name == "auto":
        name = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "local"
    if name == "anthropic":
        return AnthropicBackend()
    if name == "local":
        return LocalBackend()
    if name in llm_providers.PROVIDERS:
        return OpenAICompatibleBackend(name)
    raise LLMUnavailable(
        f"unknown LLM_BACKEND {name!r}; expected one of: local, anthropic, "
        f"none, " + ", ".join(sorted(llm_providers.PROVIDERS)))


def get_backend():
    """Cached backend, or None if the stage is unavailable.

    A failure is **not** cached. An earlier version stored False on the first
    error, so a single transient out-of-memory disabled the LLM for the life of
    the process -- every later request reported "unavailable" on a machine where
    the model loaded fine. Construction is cheap relative to a failed request,
    so retrying per call is the right trade.

    An explicit `LLM_BACKEND=none` is cached, because that is a decision rather
    than a failure.
    """
    global _BACKEND, _LAST_ERROR
    if _BACKEND is not None:
        return _BACKEND or None
    try:
        backend = resolve_backend()
        _LAST_ERROR = None
    except LLMUnavailable as exc:
        # Keep the reason. An earlier version discarded it, so the pipeline
        # reported a bare "unavailable" for three separate root causes in a row
        # and each took a fresh investigation to identify.
        _LAST_ERROR = str(exc)
        return None
    if backend is None:
        _BACKEND = False   # deliberately disabled
        return None
    _BACKEND = backend
    return backend


def fallback_backends(primary):
    """Backends to try after `primary`, in preference order.

    Free tiers are the point of this project's defaults, and a free tier is
    exactly the kind of thing that returns 429 for an hour. The provider layer
    already retries one endpoint; this is the next level up -- if NVIDIA is out
    of quota and a Groq key is sitting in the same `.env`, the stage should use
    it rather than report itself unavailable.

    Set `LLM_FALLBACK_BACKENDS` to a comma-separated list to control the order,
    or to an empty string to switch failover off entirely.
    """
    raw = os.environ.get("LLM_FALLBACK_BACKENDS")
    if raw is not None:
        names = [n.strip().lower() for n in raw.split(",") if n.strip()]
    else:
        # Any hosted provider whose key is actually present. Deriving it beats
        # asking the user to maintain a second list that duplicates their keys.
        names = [name for name, cfg in llm_providers.PROVIDERS.items()
                 if cfg.get("hosted") and cfg.get("key_env")
                 and os.environ.get(cfg["key_env"])]
    return [n for n in names if n != primary]


def _attempt(backend, transcript, context):
    """One extraction. Returns (record, error_string)."""
    try:
        record = backend.extract(transcript, **context)
    except LLMUnavailable as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return record, None


def extract_incident(transcript, **context):
    """Return (record, meta). `record` is None when the stage cannot run.

    Never raises. A failed enrichment must not fail a request that has already
    transcribed and classified a call.
    """
    if not (transcript or "").strip():
        return None, {"status": "skipped", "reason": "empty transcript"}
    backend = get_backend()
    if backend is None:
        return None, {"status": "unavailable",
                      "reason": _LAST_ERROR or "no LLM backend configured"}

    attempts = []
    record, error = _attempt(backend, transcript, context)
    attempts.append({"backend": backend.name, "error": error})

    # Only a hosted primary falls back. A local model that failed did so
    # because of this machine, and trying a second local model would fail the
    # same way after another minute of loading.
    if record is None and getattr(backend, "is_hosted", False):
        for name in fallback_backends(backend.name):
            try:
                alt = OpenAICompatibleBackend(name)
            except LLMUnavailable as exc:
                attempts.append({"backend": name, "error": str(exc)})
                continue
            record, error = _attempt(alt, transcript, context)
            attempts.append({"backend": name, "error": error})
            if record is not None:
                backend = alt
                break

    if record is None:
        return None, {"status": "failed", "backend": attempts[0]["backend"],
                      "reason": attempts[0]["error"] or "no record returned",
                      "attempts": attempts}

    try:
        # Claims must survive their own evidence. A weapon the model cannot
        # quote from the transcript is dropped here rather than reaching an
        # officer-safety advisory.
        kept, dropped = verify_evidence(record, transcript, "weapons")
        record["weapons"] = kept
        if dropped:
            record.setdefault("_rejected", {})["weapons"] = dropped
        # describe(), not getattr(backend, "model"): on LocalBackend `.model`
        # is the loaded torch module, so that expression put a 30-line
        # Qwen2ForCausalLM repr into the API response.
        meta = {"status": "ok", "backend": backend.name,
                "model": backend.describe()}
        if len(attempts) > 1:
            # The report should say the primary was skipped and why, rather
            # than quietly naming a provider the operator did not configure.
            meta["attempts"] = attempts
            meta["failed_over_from"] = attempts[0]["backend"]
        return record, meta
    except Exception as exc:
        return None, {"status": "failed", "backend": backend.name,
                      "reason": f"{type(exc).__name__}: {exc}",
                      "attempts": attempts}
