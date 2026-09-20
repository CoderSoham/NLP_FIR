"""OpenAI-compatible providers, hosted and local.

NVIDIA, Groq, OpenRouter, Cerebras, Mistral, Ollama, llama.cpp and vLLM all
speak the same `/chat/completions` shape, so one backend reaches all of them.
That matters twice over:

- **A free hosted tier removes the hardware ceiling.** This machine's GTX 1060
  cannot hold a model large enough to extract reliably; a 49B or 70B model on
  someone else's GPU can.
- **The same code reaches a local server.** Ollama and llama.cpp expose the
  same endpoint, which is how a *quantised* model runs on a Pascal card that
  bitsandbytes will not support. No cloud, no key, no torch.

Written against `urllib` rather than a client library: it is one POST with a
JSON body, and this project has already been broken once by an unconstrained
`pip install`.

> **Sending a transcript to a hosted provider sends emergency-call content —
> names, addresses, phone numbers, medical detail — to a third party.** Several
> free tiers train on submitted prompts. That is why no hosted provider is ever
> the default, and why `local` and `ollama` exist alongside them.
"""
import json
import os
import time
import urllib.error
import urllib.request

# Some providers sit behind a CDN that rejects urllib's default agent
# ("Python-urllib/3.12") outright -- Groq returns Cloudflare error 1010,
# an HTTP 403 that looks like an auth failure and is not.
USER_AGENT = "nlp-fir/1.0 (+https://github.com/CoderSoham/NLP_FIR)"


PROVIDERS = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_env": "NVIDIA_API_KEY",
        # Measured 53/54 on the nine-sample evaluation set at a 21s median --
        # the best of every model tried, hosted or local. The previous default
        # here was end-of-life and returned HTTP 410, which is why the CLI
        # queries /models rather than trusting any hardcoded id.
        "default_model": "nvidia/nemotron-3-ultra-550b-a55b",
        "hosted": True,
        "note": "build.nvidia.com. Free permanent key, no card, ~40 req/min.",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "hosted": True,
        "note": "Fastest hosted free tier; lower daily token allowance.",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
        "hosted": True,
        "note": "Many models behind one key; ':free' variants are rate limited.",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "key_env": "CEREBRAS_API_KEY",
        "default_model": "llama-3.3-70b",
        "hosted": True,
        "note": "High daily token allowance; card required since mid-2026.",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "key_env": "MISTRAL_API_KEY",
        "default_model": "mistral-large-latest",
        "hosted": True,
        "note": "Free experimentation tier.",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
        "default_model": "qwen2.5:7b-instruct-q4_K_M",
        "hosted": False,
        "note": "Local. Runs quantised models on GPUs transformers cannot.",
    },
    "llamacpp": {
        "base_url": "http://localhost:8080/v1",
        "key_env": None,
        "default_model": "local-model",
        "hosted": False,
        "note": "Local llama.cpp server. Same endpoint shape as Ollama.",
    },
}


def resolve(name):
    """Return the provider config, with env overrides applied."""
    if name not in PROVIDERS:
        raise KeyError(name)
    cfg = dict(PROVIDERS[name])
    cfg["name"] = name
    cfg["base_url"] = os.environ.get("LLM_BASE_URL") or cfg["base_url"]
    cfg["model"] = os.environ.get("LLM_MODEL") or cfg["default_model"]
    cfg["api_key"] = os.environ.get(cfg["key_env"]) if cfg["key_env"] else None
    return cfg


def chat_completion(cfg, messages, timeout=None, json_mode=True,
                    max_tokens=None, temperature=0.0):
    """One /chat/completions call. Returns the assistant's text.

    `json_mode` asks for a JSON object where the provider supports it. Not all
    do, and an unsupported field is rejected rather than ignored by some -- so
    a 400 mentioning response_format retries without it instead of failing the
    stage.
    """
    # Reasoning models emit their working before the answer, so a budget sized
    # for the JSON alone truncates them mid-object -- observed as
    # "unterminated JSON object" from nemotron-3-super. Hosted latency varies
    # far more than local, so the timeout is generous and configurable.
    max_tokens = max_tokens or int(os.environ.get("LLM_MAX_TOKENS", 8192))
    timeout = timeout or int(os.environ.get("LLM_TIMEOUT", 300))
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    # 429 is rate limiting and 5xx is shared-capacity contention; both are
    # transient and were being recorded as model failures, which understated
    # every hosted model in the evaluation.
    retries = int(os.environ.get("LLM_RETRIES", 4))
    attempt = 0
    while True:
        attempt += 1
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json",
                   "User-Agent": USER_AGENT}
        if cfg.get("api_key"):
            headers["Authorization"] = f"Bearer {cfg['api_key']}"
        request = urllib.request.Request(
            cfg["base_url"].rstrip("/") + "/chat/completions",
            data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode())
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if json_mode and "response_format" in detail and "response_format" in payload:
                payload.pop("response_format")
                continue
            if exc.code in (408, 409, 429, 500, 502, 503, 504) and attempt <= retries:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise RuntimeError(f"HTTP {exc.code} from {cfg['name']}: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt <= retries:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise RuntimeError(f"cannot reach {cfg['name']} at "
                               f"{cfg['base_url']}: {exc.reason}") from exc


def list_models(cfg, timeout=30):
    """Ask the provider what it actually serves.

    Model identifiers drift and vary between catalogues -- the same Nemotron
    appears as `nemotron-3-super-120b-a12b` in one place and with a doubled
    vendor prefix in another. Querying `/models` is the only way to be right,
    and it costs one request.
    """
    headers = {"User-Agent": USER_AGENT}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    request = urllib.request.Request(
        cfg["base_url"].rstrip("/") + "/models", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"HTTP {exc.code} listing models for {cfg['name']}: "
            f"{exc.read().decode(errors='replace')[:200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach {cfg['name']}: {exc.reason}") from exc
    return sorted(m.get("id", "") for m in data.get("data", []) if m.get("id"))


# Substrings that suggest a model is a poor fit for this task, used only to
# order the suggestions -- never to hide anything the provider offers.
_UNSUITABLE = ("embed", "rerank", "guard", "vision", "ocr", "vila", "clip",
               "speech", "asr", "tts", "riva", "diffusion", "sana", "flux")


def suggest_models(model_ids, limit=12):
    """Order a catalogue by plausible fit for structured extraction.

    Preference is for large instruct-tuned text models. The task is reading a
    noisy transcript and filling a schema, so reasoning and instruction
    following matter and modality does not.
    """
    def rank(mid):
        low = mid.lower()
        if any(bad in low for bad in _UNSUITABLE):
            return (3, mid)
        score = 2
        if any(good in low for good in ("nemotron", "llama", "qwen", "deepseek",
                                        "mistral", "glm", "kimi")):
            score = 0
        elif "instruct" in low or "chat" in low:
            score = 1
        return (score, mid)

    return sorted(model_ids, key=rank)[:limit]


def describe_providers():
    """Human-readable list, for the CLI helper and the README."""
    lines = []
    for name, cfg in PROVIDERS.items():
        where = "hosted" if cfg["hosted"] else "local "
        key = cfg["key_env"] or "no key"
        lines.append(f"  {name:12} [{where}] {key:20} {cfg['default_model']}")
        lines.append(f"  {'':12}   {cfg['note']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        provider = sys.argv[1]
        cfg = resolve(provider)
        if cfg["key_env"] and not cfg["api_key"]:
            print(f"Set {cfg['key_env']} first:  export {cfg['key_env']}=...")
            raise SystemExit(1)
        models = list_models(cfg)
        print(f"{provider}: {len(models)} models at {cfg['base_url']}\n")
        print("Most likely to suit transcript extraction:")
        for m in suggest_models(models):
            print("  ", m)
        print(f"\nAll ids: run with --all to print every one")
        if "--all" in sys.argv:
            for m in models:
                print("  ", m)
        raise SystemExit(0)

    print("LLM_BACKEND options beyond 'local', 'anthropic' and 'none':\n")
    print(describe_providers())
    print("\nHosted providers receive the transcript. Emergency-call transcripts")
    print("contain names, addresses and medical detail, and some free tiers")
    print("train on submitted prompts. Use 'local' or 'ollama' to keep them here.")
