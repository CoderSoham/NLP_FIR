# 911 First Responder – AI-Powered Voice Triage and FIR Report Generator

A production-ready Flask web application that intelligently processes emergency call audio through a multi-stage AI/ML pipeline. The system transcribes calls using OpenAI Whisper, performs advanced NLP analysis (emergency type classification, severity assessment, sentiment/emotion detection), extracts entities and probable locations, generates actionable dispatch recommendations, and exports a comprehensive FIR-style PDF report. Designed to support emergency response teams with rapid call triage and decision support.

---

## 🎯 Key Features

| Feature | Details |
|---------|---------|
| **Multi-Format Audio Upload** | Supports mp3, wav, ogg, m4a; max 25 MB with server-side trimming to ~120s for latency optimization |
| **Automatic Transcription** | OpenAI Whisper with language auto-detection and optional translation to English for analysis |
| **Advanced NLP Analysis** | <ul><li>Emergency type classification (zero-shot + contextual)</li><li>Severity assessment (zero-shot + sentiment + emotion)</li><li>Named entity recognition (spaCy)</li><li>Probable location extraction (GPE/LOC/FAC entities)</li><li>Command template matching (sentence-transformers)</li><li>Abstractive summarization (T5)</li></ul> |
| **Intelligent Dispatch** | Nearest station heuristic with ETA calculation and required resources recommendation |
| **Rich Visualizations** | Waveform display, MFCC spectrograms, pitch contours |
| **FIR Report Export** | Structured PDF report with transcription, analysis, recommendations, and metadata |
| **RESTful JSON API** | For programmatic integration with dispatch systems or third-party apps |
| **Health Monitoring** | Built-in `/healthz` endpoint for deployment monitoring |

---

## 📁 Project Structure

```
911FirstResponder/
├── app.py                          # Flask application (routes + API endpoints)
├── config.py                       # Centralized configuration & environment setup
├── test.py                         # Standalone testing script
├── requirements.txt                # Python dependencies (pinned versions)
├── README.md                       # This file
│
├── utils/
│   ├── audio_utils.py              # Core inference pipeline
│   │                               # - Audio loading & preprocessing
│   │                               # - Whisper transcription
│   │                               # - NLP analysis (classification, NER, sentiment)
│   │                               # - Plot generation (waveform, MFCC, pitch)
│   │                               # - FIR PDF generation
│   │                               # - Dispatch suggestion logic
│   └── custom_model_class.py       # Placeholder for custom STT model integration
│
├── templates/
│   ├── index.html                  # Main web interface
│   └── result.html                 # Legacy template (not currently used)
│
├── static/
│   ├── css/
│   │   └── styles.css              # Application styling
│   ├── plots/                      # Generated plot images (waveform, MFCC, pitch)
│   ├── audio/                      # Sample audio files (dev/testing)
│   └── uploads/                    # User-uploaded audio files
│
├── models/
│   ├── transcription_tensor.pth    # Optional pre-trained model weights
│   └── transcriptions_final.csv    # Sample transcription reference data
│
├── processed/                      # Output directory for generated reports
│   └── fir_report.pdf              # Generated FIR PDF report
│
└── plots/                          # Intermediate generated plots
```

---

## ⚙️ How It Works: End-to-End Flow

```
User Audio Upload
       ↓
┌──────────────────────────────────────┐
│ 1. FILE INTAKE & VALIDATION          │
│  • Save with UUID filename           │
│  • Verify file format (mp3/wav/...)  │
│  • Check file size (<25 MB)          │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│ 2. AUDIO PREPROCESSING               │
│  • Load at 16 kHz mono               │
│  • Trim to ≤120 seconds              │
│  • Generate visualizations:          │
│    - Waveform plot                   │
│    - MFCC spectrogram                │
│    - Pitch contour                   │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│ 3. SPEECH-TO-TEXT                    │
│  • Whisper transcription             │
│  • Language detection                │
│  • If non-English → Translate to EN  │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│ 4. NLP ANALYSIS (on English text)    │
│                                      │
│  A. Emergency Type Classification    │
│     • Zero-shot classifier           │
│     • Context-aware categorization   │
│     → Types: Medical, Fire, Traffic, │
│        Police, Disaster, Other       │
│                                      │
│  B. Severity Assessment              │
│     • Zero-shot scoring              │
│     • Sentiment analysis             │
│     • Emotion detection              │
│     • Entity-based heuristics        │
│     → Levels: Critical, High,        │
│        Medium, Low                   │
│                                      │
│  C. Named Entity Recognition         │
│     • spaCy NER (en_core_web_sm)     │
│     • Extract: Person, Location,     │
│       Organization, Facility         │
│                                      │
│  D. Location Extraction              │
│     • Identify GPE/LOC/FAC entities  │
│     • Heuristic for "likely location"│
│                                      │
│  E. Command Template Matching        │
│     • Sentence-Transformer embeddings│
│     • Cosine similarity to templates │
│                                      │
│  F. Abstractive Summarization        │
│     • T5 model (fine-tuned context)  │
│     • 100-word emergency summary     │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│ 5. DISPATCH & RECOMMENDATIONS        │
│  • Extract required resources        │
│  • Calculate nearest station (ETA)   │
│  • Generate action recommendations   │
│  • Priority level assignment         │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│ 6. REPORT GENERATION                 │
│  • Create structured FIR PDF:        │
│    - Header (timestamp, call ID)     │
│    - Transcription                   │
│    - Analysis results                │
│    - Recommended actions             │
│    - Required resources              │
│    - Dispatch suggestion + ETA       │
│  • Save to processed/fir_report.pdf  │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│ 7. RESPONSE & DELIVERY               │
│  • Return JSON with all results      │
│  • Display on web UI                 │
│  • Allow PDF download                │
│  • Store for audit trail             │
└──────────────────────────────────────┘
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.8+** (tested with 3.9, 3.10, 3.11)
- **pip** or **conda** for package management
- **Git** (optional, for cloning)
- **~4-5 GB disk space** for model weights (Whisper, spaCy, transformers, sentence-transformers)

### Step 1: Clone or Extract the Repository

```bash
# If using git
git clone https://github.com/yourusername/911FirstResponder.git
cd 911FirstResponder

# Or extract the zip/tar file
cd 911FirstResponder
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Using venv (built-in)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n responder python=3.10
conda activate responder
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** First-time installation will download and cache:
- Whisper model (base: ~140 MB)
- spaCy en_core_web_sm (~40 MB)
- Sentence-Transformers (~80 MB)
- Transformers & pre-trained models (~2-3 GB)

This is a one-time operation and may take 5-15 minutes depending on connection speed.

### Step 4: Create Required Directories

```bash
mkdir -p static/uploads static/plots processed plots
```

### Step 5: Run the Application

```bash
# Development server
python app.py

# Or with Gunicorn (production)
gunicorn -w 4 -b 0.0.0.0:5000 app.py
```

The app will be available at `http://localhost:5000`

---

## 📋 Configuration

All settings are controlled via `config.py` and environment variables:

```python
# Folders
UPLOAD_FOLDER = 'static/uploads'           # Uploaded audio storage
PROCESSED_FOLDER = 'processed'             # Output reports/plots
STATIC_PLOTS_FOLDER = 'static/plots'       # Public plot copies

# Upload constraints
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg'}
MAX_CONTENT_LENGTH = 25 * 1024 * 1024      # 25 MB max file size

# App settings
DEBUG = False                              # Flask debug mode
SECRET_KEY = 'change-this-in-prod'        # Session/CSRF secret

# Model settings
WHISPER_MODEL_NAME = 'base'                # 'tiny', 'small', 'base', 'medium', 'large'
FORCE_CPU = True                           # Force CPU even with GPU available
TOKENIZERS_PARALLELISM = 'false'           # Prevent parallelism warnings
```

### Environment Variable Overrides

```bash
export WHISPER_MODEL_NAME=small
export FORCE_CPU=0  # Use GPU if available
export MAX_CONTENT_LENGTH=$((50 * 1024 * 1024))  # 50 MB
export FLASK_DEBUG=1
python app.py
```

---

## 🔌 API Endpoints

### 1. Web Interface (GET / POST)

**URL:** `GET/POST /`

Upload audio via web form, returns rendered HTML with results.

**Example:**
```bash
curl -F "audio_file=@call.mp3" http://localhost:5000/
```

### 2. JSON API

**URL:** `POST /api/process`

Submit audio file and receive structured JSON response.

**Request:**
```bash
curl -X POST -F "audio_file=@call.mp3" http://localhost:5000/api/process
```

**Response:**
```json
{
  "audio_filename": "a1b2c3d4e5f6.mp3",
  "audio_url": "/audio/a1b2c3d4e5f6.mp3",
  "transcription": "There's a car accident on Main Street at 5th Avenue...",
  "detected_language": "en",
  "emergency_type": {
    "type": "Traffic Accident",
    "confidence": 0.98
  },
  "severity": {
    "level": "High",
    "score": 0.85
  },
  "sentiment": "negative",
  "emotion": "urgency",
  "entities": [
    {"text": "Main Street", "label": "GPE"},
    {"text": "5th Avenue", "label": "FAC"}
  ],
  "probable_location": "Main Street at 5th Avenue",
  "summary": "Traffic accident reported on Main Street at 5th Avenue with potential injuries...",
  "recommended_actions": [
    "Dispatch ambulance to location",
    "Alert traffic control units",
    "Notify hospital emergency department"
  ],
  "required_resources": ["Ambulance", "Police", "Traffic Control"],
  "dispatch_suggestion": {
    "nearest_station": "Central Fire Station",
    "eta_minutes": 8,
    "priority": "High"
  },
  "plots": {
    "waveform": "/static/plots/waveform.png",
    "mfcc": "/static/plots/mfcc.png",
    "pitch": "/static/plots/pitch.png"
  },
  "fir_pdf_path": "/download_fir",
  "timestamp": "2026-01-28T14:32:15"
}
```

### 3. Health Check

**URL:** `GET /healthz`

```bash
curl http://localhost:5000/healthz
# Response: {"status": "ok"}
```

### 4. Download FIR Report

**URL:** `GET /download_fir`

```bash
curl http://localhost:5000/download_fir -o fir_report.pdf
```

### 5. Serve Uploaded Audio

**URL:** `GET /audio/<filename>`

```bash
curl http://localhost:5000/audio/a1b2c3d4e5f6.mp3 > call.mp3
```

---

## 🧠 ML Models & Components

| Component | Model | Purpose | Device |
|-----------|-------|---------|--------|
| **Transcription** | OpenAI Whisper (base/small/medium/large) | Speech-to-text with language detection | CPU/GPU |
| **Emergency Classification** | Facebook BART-large-MNLI | Zero-shot emergency type classification | CPU |
| **Severity Assessment** | Microsoft DeBERTa-v3 | Zero-shot severity scoring | CPU |
| **Named Entity Recognition** | spaCy en_core_web_sm | Extract persons, locations, organizations | CPU |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) | Command template matching | CPU |
| **Summarization** | T5-small | Abstractive emergency summary | CPU |
| **Sentiment Analysis** | transformers pipeline | Sentiment classification | CPU |
| **Emotion Detection** | transformers (custom fine-tuned) | Emotion in audio context | CPU |

---

## 📊 Audio Feature Extraction

The system extracts and visualizes:

1. **Waveform** - Time-domain audio signal
2. **MFCC (Mel-Frequency Cepstral Coefficients)** - Frequency representation (13 coefficients)
3. **Pitch Contour** - Fundamental frequency over time (using librosa piptrack)

All plots are saved as PNG images in `static/plots/` and `processed/` for archival.

---

## 🔄 Development Workflow

### Running Tests

```bash
python test.py
```

This script:
- Prompts for MP3 file path
- Loads models and processes audio
- Generates plots and extracts features
- Displays named entities

### Custom Model Integration

To use a custom STT model instead of Whisper:

1. Implement your model in [utils/custom_model_class.py](utils/custom_model_class.py)
2. Update [utils/audio_utils.py](utils/audio_utils.py) to import and use your model
3. Modify `load_whisper_model()` or add a new loader function

---

## ⚡ Performance Tuning

### For Faster Processing

```bash
# Use smaller Whisper model
export WHISPER_MODEL_NAME=tiny  # or 'small'

# Reduce model precision (if using GPU)
export TORCH_DTYPE=float16
```

### For Better Accuracy

```bash
# Use larger Whisper model
export WHISPER_MODEL_NAME=large
```

### For GPU Acceleration

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Disable CPU-only mode
export FORCE_CPU=0
```

---

## 🐳 Deployment Options

### Docker (Recommended for Production)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app.py"]
```

Build & run:
```bash
docker build -t 911-responder .
docker run -p 5000:5000 911-responder
```

### Cloud Deployment (AWS, GCP, Azure)

For cloud platforms, ensure:
1. Sufficient disk space for model caching
2. Appropriate memory (4+ GB recommended)
3. Persistent storage for `processed/` and `static/uploads/`
4. Optional GPU for faster Whisper processing

---

## 🔒 Security Considerations

- **Input Validation:** File type and size checking enabled
- **Filename Sanitization:** UUID-based filenames prevent path traversal
- **CORS:** Configure as needed for production
- **Secret Key:** Change `SECRET_KEY` in production
- **Cleanup:** Implement periodic cleanup of `static/uploads/` and `processed/`

**Recommendations:**
- Use HTTPS in production
- Implement authentication for API endpoints
- Add request rate limiting
- Sanitize and validate all user inputs
- Run behind a reverse proxy (nginx, Apache)

---

## 📝 Logging & Debugging

Enable verbose logging:

```bash
export FLASK_DEBUG=1
python app.py
```

Check logs for:
- Model loading status
- API request/response timing
- NLP analysis confidence scores
- Error messages with stack traces

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Multi-language support for analysis
- [ ] Real-time WebSocket streaming for live calls
- [ ] Integration with actual dispatch systems
- [ ] Fine-tuned models for emergency-specific vocabulary
- [ ] Database backend for call history
- [ ] Advanced visualization dashboards
- [ ] Performance benchmarking suite

---

## 📄 License

[Add your license here - MIT, Apache 2.0, etc.]

---

## 🆘 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Set `FORCE_CPU=1` or use a smaller Whisper model

### Issue: "Model download stuck"
**Solution:** Download models manually:
```bash
python -c "import whisper; whisper.load_model('base')"
python -m spacy download en_core_web_sm
```

### Issue: "Audio file not supported"
**Solution:** Convert to wav/mp3 first:
```bash
ffmpeg -i input.m4a output.mp3
```

### Issue: "Plots not generating"
**Solution:** Ensure directories exist:
```bash
mkdir -p static/plots processed plots
chmod 755 static/plots processed plots
```

---

## 📞 Support & Contact

For issues, questions, or suggestions:
- Open an GitHub issue
- Contact: [your-email@example.com]
- Documentation: [link to docs]

---

## 🎓 References & Resources

- [OpenAI Whisper](https://github.com/openai/whisper)
- [spaCy NLP Library](https://spacy.io/)
- [Sentence-Transformers](https://www.sbert.net/)
- [librosa Audio Analysis](https://librosa.org/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

**Last Updated:** January 28, 2026  
**Version:** 1.0.0  
**Status:** Production Ready ✅

## Setup

### Prerequisites

- Windows 10/11 with Python 3.10 or 3.11
- ffmpeg installed and on PATH (required by librosa/pydub)
  - Easiest: `choco install ffmpeg -y` (Chocolatey) or download static build and add `bin` to PATH

### Install Dependencies

You can use your existing Python environment (no venv required), but a venv is recommended in general.

```
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # Alternative below if this fails
```

If the `spacy download` command fails due to network restrictions, install the wheel directly:

```
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

Note: The requirements pin NumPy (1.26.4) and Numba (<0.60) to keep Whisper compatible on Windows.

### Run the App

```
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

Health check: `http://127.0.0.1:5000/healthz`

### Environment Configuration

Set environment variables to tune behavior (optional):

- `WHISPER_MODEL_NAME` → default: `base` (try `tiny`, `small` for speed/accuracy trade-offs)
- `UPLOAD_FOLDER` → default: `static/uploads`
- `PROCESSED_FOLDER` → default: `processed`
- `STATIC_PLOTS_FOLDER` → default: `static/plots`
- `MAX_CONTENT_LENGTH` → default: `26214400` (25 MB)
- `FLASK_DEBUG` → `1` to enable debug during development
- `TOKENIZERS_PARALLELISM` → `false` to silence HF tokenizer warnings

PowerShell example:

```
$env:WHISPER_MODEL_NAME="tiny"
$env:FLASK_DEBUG="1"
python app.py
```

## JSON API

`POST /api/process`

Form-data: `audio_file=@<path-to-audio>`

Response JSON includes:
`transcription`, `translated_text`, `language`, `emergency_type`, `severity`, `response`, `response_time`, `summary`, `entities`, `best_match`, `score`, `sentiment`, `emotion`, `probable_location`, `dispatch`.

Example (PowerShell):

```
curl --% -X POST http://127.0.0.1:5000/api/process -F "audio_file=@static\audio\call_9.mp3"
```

## Notes on Models and Performance

- Default is CPU inference. For faster runs, set `WHISPER_MODEL_NAME=tiny` and keep audio short.
- The zero-shot and summarization pipelines are heavier; on small machines they may add latency. We already run under `torch.inference_mode()` where applicable.
- Consider moving long inference to a background worker (Celery/RQ) for production.

## Roadmap

- Recent-calls dashboard and persistence
- Background job queue for long audio
- Resource-aware dispatch backed by geodata and agency rosters
- Streaming transcription (WebSocket)
- PII redaction and retention policies

## Troubleshooting

- Whisper/Numba import errors: ensure `numpy==1.26.4` and `numba<0.60`
- spaCy model not found: install `en_core_web_sm` via `spacy download` or direct wheel
- Tokenizer errors: `pip install sentencepiece` (already included in `requirements.txt`)
- ffmpeg not found: add to PATH or install via Chocolatey

## License

For internal evaluation; add a license if distributing.
