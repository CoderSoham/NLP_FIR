"""Speech recognition via faster-whisper (CTranslate2).

Chosen over `openai-whisper` because CTranslate2 runs the same weights several
times faster with int8 quantisation, and that speed is what makes a *large*
model affordable. Accuracy on noisy telephone audio is dominated by model size,
so the fastest path to better transcripts is a bigger model, not a better
decoder.

Device and quantisation are resolved from the hardware rather than assumed:
a GTX 1060 is Pascal and CTranslate2 offers it `{int8, float32, int8_float32}`
with no float16, so hardcoding `float16` -- the value in most faster-whisper
examples -- fails on exactly the card this was developed on.
"""
import os

from utils import gpu
from utils.asr_cloud import (audio_fingerprint, cache_get, cache_put,
                             transcribe_groq)
from utils.cuda_libs import cuda_usable

_MODEL = None
_MODEL_KEY = None
_DEGRADED = None

# Domain vocabulary. Whisper accepts an initial prompt as decoding context; on
# emergency calls it measurably improves recognition of dispatcher phrasing and
# the words that matter most downstream.
DEFAULT_PROMPT = (
    "911, what is your emergency? Dispatcher and caller on a recorded emergency "
    "line. Ambulance, paramedics, police officer, fire department, engine "
    "company. Address, cross street, apartment number, callback number. "
    "Suspect, weapon, firearm, gunshot, stabbing, assault, domestic. "
    "Unconscious, not breathing, CPR, chest pain, seizure, overdose, bleeding. "
    "Structure fire, smoke, gas leak, hazmat. Motor vehicle collision, rollover."
)


def resolve_device(force_cpu=None):
    """(device, compute_type) that this machine can actually run."""
    if force_cpu is None:
        force_cpu = os.environ.get("FORCE_CPU", "1") == "1"

    requested = os.environ.get("ASR_COMPUTE_TYPE")
    if not force_cpu and cuda_usable():
        try:
            import ctranslate2
            supported = ctranslate2.get_supported_compute_types("cuda")
        except Exception:
            supported = {"int8"}
        if requested and requested in supported:
            return "cuda", requested
        for candidate in ("float16", "int8_float16", "int8_float32", "int8"):
            if candidate in supported:
                return "cuda", candidate
        return "cuda", "int8"

    return "cpu", requested or "int8"


def default_model_name(device):
    """A model size that is sane for the device, unless one is configured."""
    configured = os.environ.get("WHISPER_MODEL_NAME")
    if configured:
        return configured
    # turbo is ~8x faster than large-v3 at close to the same accuracy, which is
    # what makes a large model viable on a 6 GB card. On CPU it is still slow,
    # so a smaller default keeps a CPU-only install usable.
    return "large-v3-turbo" if device == "cuda" else "base"


def unload():
    """Release the ASR model's GPU memory. Returns True if anything was freed."""
    global _MODEL, _MODEL_KEY
    if _MODEL is None:
        return False
    was_cuda = bool(_MODEL_KEY and _MODEL_KEY[1] == "cuda")
    _MODEL, _MODEL_KEY = None, None
    return was_cuda


gpu.register("asr", unload)


def load_model(model_name=None, force_cpu=None):
    """Load and cache the ASR model. Falls back to CPU if the GPU refuses."""
    global _MODEL, _MODEL_KEY
    from faster_whisper import WhisperModel

    device, compute_type = resolve_device(force_cpu)
    if device == "cuda":
        gpu.acquire("asr")
    model_name = model_name or default_model_name(device)
    key = (model_name, device, compute_type)

    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL

    _MODEL = _construct(WhisperModel, model_name, device, compute_type)
    _MODEL_KEY = (model_name, device, compute_type)
    return _MODEL


def _construct(WhisperModel, model_name, device, compute_type):
    """Build the model, degrading to CPU only as a last resort.

    The fallback keeps **the same model**. An earlier version dropped to `base`
    on CPU, which is a large accuracy loss applied silently -- the exact failure
    mode this project has been bitten by repeatedly. `large-v3-turbo` on CPU is
    slower but still far better than `base`, and `degraded` records that it
    happened so the caller can surface it.
    """
    global _DEGRADED
    _DEGRADED = None
    if device != "cuda":
        return WhisperModel(model_name, device=device, compute_type=compute_type)

    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as first:
        # Something else may hold the card. Evict and try once more before
        # giving up on the GPU entirely.
        gpu.acquire("asr")
        gpu.empty_cache()
        try:
            return WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as second:
            _DEGRADED = (f"GPU unavailable ({type(second).__name__}: "
                         f"{str(second).splitlines()[0][:120]}); running on CPU")
            return WhisperModel(model_name, device="cpu", compute_type="int8")


def transcribe(audio, language=None, translate=False, initial_prompt=None,
               beam_size=5, model=None):
    """Transcribe (or translate to English). Returns a result dict.

    `vad_filter` drops non-speech before decoding. That matters more than it
    sounds: Whisper hallucinates confident text over silence and hold music,
    and emergency calls contain a lot of both.

    `condition_on_previous_text=False` stops a bad segment from poisoning
    everything after it, which is the failure mode behind Whisper's repetition
    loops on noisy audio.
    """
    model = model or load_model()
    try:
        return _run(model, audio, language, translate, initial_prompt, beam_size)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        # Another component grew into the card between load and inference.
        # Evict, clear, rebuild, and try once. Only then fall back to CPU.
        global _MODEL, _MODEL_KEY
        _MODEL, _MODEL_KEY = None, None
        gpu.acquire("asr")
        gpu.empty_cache()
        model = load_model()
        return _run(model, audio, language, translate, initial_prompt, beam_size)


def _run(model, audio, language, translate, initial_prompt, beam_size):
    segments, info = model.transcribe(
        audio,
        language=language,
        task="translate" if translate else "transcribe",
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        initial_prompt=initial_prompt if initial_prompt is not None else DEFAULT_PROMPT,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )

    collected = []
    for segment in segments:
        collected.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
            "avg_logprob": round(segment.avg_logprob, 3),
            "no_speech_prob": round(segment.no_speech_prob, 3),
        })

    text = " ".join(s["text"] for s in collected).strip()
    confidences = [s["avg_logprob"] for s in collected]
    return {
        "text": text,
        "segments": collected,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "mean_logprob": round(sum(confidences) / len(confidences), 3) if confidences else None,
        "model": _MODEL_KEY[0] if _MODEL_KEY else None,
        "device": _MODEL_KEY[1] if _MODEL_KEY else None,
        "compute_type": _MODEL_KEY[2] if _MODEL_KEY else None,
        "degraded": _DEGRADED,
    }


def transcribe_file(audio_path, use_cache=True, backend=None, audio=None,
                    sr=16000, **kwargs):
    """Transcribe a file, reusing a cached transcript where one exists.

    This is the entry point the pipeline should use. `transcribe()` remains the
    array-level call for cases where the audio is already in memory.

    The cache is keyed on the audio's **content** hash and the model name, so
    renaming a file does not miss the cache and changing the model does not hit
    a stale one. `use_cache=False` forces a fresh transcription, which is what
    you want when the thing under test *is* the transcription.
    """
    backend = backend or os.environ.get("ASR_BACKEND", "local")
    model = (os.environ.get("GROQ_ASR_MODEL", "whisper-large-v3-turbo")
             if backend == "groq"
             else default_model_name(resolve_device()[0]))

    fingerprint = audio_fingerprint(audio_path)
    if use_cache:
        cached = cache_get(fingerprint, f"{backend}:{model}")
        if cached:
            cached["cached"] = True
            return cached

    if backend == "groq":
        result = transcribe_groq(audio_path, model=model, **kwargs)
    else:
        if audio is None:
            import librosa
            from utils.audio_clean import clean_audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            # Same conditioning the pipeline applies, so a cached transcript
            # and a pipeline transcript of the same file are the same thing.
            audio, _report = clean_audio(audio, sr)
        result = transcribe(audio, **kwargs)

    result["cached"] = False
    result["backend"] = backend
    result["fingerprint"] = fingerprint
    cache_put(fingerprint, f"{backend}:{model}", result)
    return result
