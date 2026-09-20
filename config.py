import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load a project-local .env before anything reads the environment.
# `python-dotenv` was already a declared dependency and nothing called it, so a
# .env file sat there being ignored. Real environment variables win over the
# file, which is what you want when a container or CI supplies them.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)
except ImportError:
    pass

# Folders
# Uploads live outside static/. Serving them from the static mount made every
# recording publicly fetchable by filename, forever.
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', os.path.join('storage', 'uploads'))
PROCESSED_FOLDER = os.environ.get('PROCESSED_FOLDER', os.path.join('storage', 'processed'))

# Upload constraints
ALLOWED_EXTENSIONS = { 'wav', 'mp3', 'm4a', 'ogg' }
MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 25 * 1024 * 1024))  # 25 MB

# Retention. Emergency-call audio is not something to keep indefinitely, so
# uploads and generated artefacts are deleted once they age out.
RETENTION_SECONDS = int(os.environ.get('RETENTION_SECONDS', 3600))
RETENTION_SWEEP_SECONDS = int(os.environ.get('RETENTION_SWEEP_SECONDS', 300))

# Rate limiting. Each request costs tens of seconds of CPU, so the unit worth
# limiting is "pipeline runs started", not bytes or connections.
RATE_LIMIT_REQUESTS = int(os.environ.get('RATE_LIMIT_REQUESTS', 10))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get('RATE_LIMIT_WINDOW_SECONDS', 600))

# Only consult X-Forwarded-For when a proxy you control sets it. Off by
# default: trusting it unconditionally lets any caller reset their own bucket.
TRUST_PROXY_HEADERS = os.environ.get('TRUST_PROXY_HEADERS', '0') == '1'

# Unset means open. Set it and the upload routes require X-API-Key.
API_KEY = os.environ.get('API_KEY', '')

# App settings
DEBUG = os.environ.get('FLASK_DEBUG', '0') == '1'

# Whisper shells out to ffmpeg by name, so it must be on PATH before the first
# transcription. Done here because config is imported before utils.audio_utils.
from utils.ffmpeg_path import ensure_ffmpeg_on_path
FFMPEG_PATH = ensure_ffmpeg_on_path()

# Audio is truncated to bound worst-case request time; the result reports how
# much of the call was actually analysed.
#
# The cap exists because local transcription is roughly linear in duration and
# one long recording would otherwise occupy the only worker for minutes. Hosted
# transcription does not have that problem -- a ten-minute call comes back in
# seconds -- so the cap is far higher when it is in use. Truncating a 696-second
# call to 120 seconds throws away the outcome of the call, which is usually the
# part that matters.
_DEFAULT_MAX_AUDIO = 1800 if os.environ.get('ASR_BACKEND') == 'groq' else 120
MAX_AUDIO_SECONDS = int(os.environ.get('MAX_AUDIO_SECONDS', _DEFAULT_MAX_AUDIO))

# The emotion detector emits 7 classes, so chance is ~0.14. Below this the label
# carries no information and must not move the severity score.
EMOTION_MIN_CONFIDENCE = float(os.environ.get('EMOTION_MIN_CONFIDENCE', 0.5))

# Model settings
WHISPER_MODEL_NAME = os.environ.get('WHISPER_MODEL_NAME', 'base')

# CPU by default. A 6 GB card will not hold both BART-large models at once, so
# opting in is a decision about your hardware, not a default we can pick.
FORCE_CPU = os.environ.get('FORCE_CPU', '1') == '1'

# Must reach HuggingFace through the environment, not as a Python name --
# transformers reads os.environ at import time. Set here so it is set before
# utils.audio_utils imports transformers.
os.environ.setdefault('TOKENIZERS_PARALLELISM',
                      os.environ.get('TOKENIZERS_PARALLELISM', 'false'))


