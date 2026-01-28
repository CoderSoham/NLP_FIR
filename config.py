import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folders
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'static/uploads')
PROCESSED_FOLDER = os.environ.get('PROCESSED_FOLDER', 'processed')
STATIC_PLOTS_FOLDER = os.environ.get('STATIC_PLOTS_FOLDER', os.path.join('static', 'plots'))

# Upload constraints
ALLOWED_EXTENSIONS = { 'wav', 'mp3', 'm4a', 'ogg' }
MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 25 * 1024 * 1024))  # 25 MB

# App settings
DEBUG = os.environ.get('FLASK_DEBUG', '0') == '1'
SECRET_KEY = os.environ.get('SECRET_KEY', 'change-this-in-prod')

# Model settings
WHISPER_MODEL_NAME = os.environ.get('WHISPER_MODEL_NAME', 'base')
FORCE_CPU = os.environ.get('FORCE_CPU', '1') == '1'
TOKENIZERS_PARALLELISM = os.environ.get('TOKENIZERS_PARALLELISM', 'false')


