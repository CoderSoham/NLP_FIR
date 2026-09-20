"""Hosted speech recognition, and a transcript cache.

Two separate ideas that belong together:

**Hosted ASR.** Groq serves `whisper-large-v3-turbo` -- the same model this
project runs locally -- on LPU hardware at a large multiple of real time. It
takes transcription off the local GPU entirely, which matters on a laptop.

**A cache.** Transcribing is the expensive half of the pipeline and the part
that changes least. Keyed on the audio's content hash plus the model, so the
same recording is never transcribed twice unless the model changes or the
caller explicitly asks. Model comparison then costs no ASR at all.

> Hosted ASR uploads **the audio itself**, not merely the transcript. For
> emergency-call recordings that is a larger disclosure than the LLM stage.
> Local remains the default.
"""
import hashlib
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid

CACHE_DIR = os.environ.get(
    "TRANSCRIPT_CACHE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "storage", "transcripts"))

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_DEFAULT_MODEL = "whisper-large-v3-turbo"

# Some providers sit behind a CDN that rejects urllib's default agent
# ("Python-urllib/3.12") outright -- Groq returns Cloudflare error 1010,
# an HTTP 403 that looks like an auth failure and is not.
USER_AGENT = "nlp-fir/1.0 (+https://github.com/CoderSoham/NLP_FIR)"



def audio_fingerprint(path_or_bytes):
    """Content hash, so the cache key survives a rename or a copy."""
    digest = hashlib.sha256()
    if isinstance(path_or_bytes, (bytes, bytearray)):
        digest.update(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()[:32]


def cache_path(fingerprint, model):
    safe = model.replace("/", "_").replace(":", "_")
    return os.path.join(CACHE_DIR, f"{fingerprint}__{safe}.json")


def cache_get(fingerprint, model):
    path = cache_path(fingerprint, model)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def cache_put(fingerprint, model, result):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(fingerprint, model)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    os.replace(tmp, path)          # atomic: a crash cannot leave a half file
    return path


def _multipart(fields, filename, content, field_name="file"):
    """Build a multipart body without pulling in a HTTP client library."""
    boundary = uuid.uuid4().hex
    parts = []
    for key, value in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                     f'name="{key}"\r\n\r\n{value}\r\n'.encode())
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                 f'name="{field_name}"; filename="{os.path.basename(filename)}"\r\n'
                 f"Content-Type: {mime}\r\n\r\n".encode())
    parts.append(content)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def transcribe_groq(audio_path, model=None, language=None, prompt=None,
                    timeout=180, retries=3):
    """Transcribe one file with Groq. Returns the same shape as utils.asr."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    model = model or os.environ.get("GROQ_ASR_MODEL", GROQ_DEFAULT_MODEL)
    with open(audio_path, "rb") as fh:
        content = fh.read()

    fields = {"model": model, "response_format": "verbose_json",
              "temperature": "0"}
    if language:
        fields["language"] = language
    if prompt:
        fields["prompt"] = prompt

    body, content_type = _multipart(fields, audio_path, content)

    last = None
    for attempt in range(retries):
        request = urllib.request.Request(
            GROQ_URL, data=body,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": content_type,
                     "User-Agent": USER_AGENT},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode())
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            last = RuntimeError(f"HTTP {exc.code} from groq: {detail}")
            # 429 is a rate limit and 5xx is capacity; both are worth waiting
            # out. 4xx otherwise means the request is wrong and will stay wrong.
            if exc.code not in (429, 500, 502, 503, 504):
                raise last from exc
            time.sleep(2 ** attempt)
        except urllib.error.URLError as exc:
            last = RuntimeError(f"cannot reach groq: {exc.reason}")
            time.sleep(2 ** attempt)
    else:
        raise last

    segments = [{
        "start": round(s.get("start", 0.0), 2),
        "end": round(s.get("end", 0.0), 2),
        "text": (s.get("text") or "").strip(),
        "avg_logprob": round(s.get("avg_logprob", 0.0), 3),
        "no_speech_prob": round(s.get("no_speech_prob", 0.0), 3),
    } for s in data.get("segments", [])]

    confidences = [s["avg_logprob"] for s in segments if s["avg_logprob"]]
    return {
        "text": (data.get("text") or "").strip(),
        "segments": segments,
        "language": data.get("language", "en"),
        "language_probability": None,          # Groq does not report one
        "mean_logprob": round(sum(confidences) / len(confidences), 3) if confidences else None,
        "model": model,
        "device": "groq",
        "compute_type": "hosted",
        "degraded": None,
        "duration": data.get("duration"),
    }
