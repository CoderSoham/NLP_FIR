# NLP_FIR — emergency call triage

Upload a recording of an emergency call. Get back a transcript, a classified
emergency type, a severity assessment, extracted entities and a probable
location, a summary, a dispatch suggestion, and a printable FIR-style PDF.

Runs entirely on CPU, no API keys, no external services.

```
audio in ──► transcribe ──► classify ──► extract ──► summarise ──► FIR PDF
```

---

## What it produces

From a single upload, one pass:

| Output | How |
|---|---|
| **Transcript** | Whisper, with automatic language detection |
| **English translation** | Whisper's `translate` task, only when the detected language isn't English |
| **Emergency type** | Zero-shot classification over `medical / fire / police / accident` |
| **Severity** | `high / medium / low` — zero-shot, weighted with call sentiment and speaker emotion |
| **Entities** | spaCy NER over the transcript |
| **Probable location** | First `GPE`, `LOC`, or `FAC` entity found |
| **Summary** | BART chunked with overlap, falling back to T5, and suppressed entirely if the result carries no information |
| **Intent match** | Nearest known dispatcher command by sentence-embedding cosine similarity |
| **Dispatch suggestion** | Station, units, and ETA — see [the limitation below](#the-dispatch-layer-is-a-stub) |
| **FIR report** | A formatted PDF, one file per request |
| **Audio plots** | Waveform, MFCC, and pitch contour |

## Pipeline

`utils/audio_utils.py:process_audio_file` is the whole flow:

1. **Load** at 16 kHz mono — what Whisper expects — and **truncate to 120 s**. Long
   uploads otherwise dominate request time; the cap is deliberate.
2. **Visualise** — waveform, MFCC, pitch, written as PNGs.
3. **Transcribe.** One pass with language auto-detect. If the language isn't
   English, a second pass with `task='translate'` produces the English text that
   every downstream stage actually consumes.
4. **Extract entities**, so severity has them to weigh.
5. **Classify type**, then **assess severity** — the zero-shot label distribution
   contributes 40%, negative sentiment 20%, distressed emotion 20%, entity
   density 20%.
6. **Summarise**, rejecting summaries that add nothing over the source.
7. **Suggest dispatch** from type and location.
8. **Render the PDF**, keyed to a per-request ID.

## Models

Everything is lazily loaded and cached in a module-level global — the first
request pays for it, the rest don't.

| Stage | Model | Approx. size |
|---|---|---|
| Transcription | `openai-whisper` (`base`, configurable) | 140 MB |
| Type + severity | `facebook/bart-large-mnli` | 1.6 GB |
| Summarisation | `facebook/bart-large-cnn` | 1.6 GB |
| Summary fallback | `t5-small` | 240 MB |
| Sentiment | `finiteautomata/bertweet-base-sentiment-analysis` | 540 MB |
| Emotion | `j-hartmann/emotion-english-distilroberta-base` | 330 MB |
| Embeddings | `all-MiniLM-L6-v2` | 90 MB |
| NER | spaCy `en_core_web_sm` | 12 MB |

**First run downloads roughly 4.5 GB** into the HuggingFace cache. Plan for it —
a cold container on a slow link takes a while before it serves a single request.
One MNLI model backs both classification stages rather than two.

## Running it

Needs Python 3.9+, ffmpeg (Whisper depends on it), and ~6 GB free disk.

```bash
git clone https://github.com/CoderSoham/NLP_FIR.git && cd NLP_FIR
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

```bash
python app.py
```

Then open `http://localhost:5000` and upload a `wav`, `mp3`, `m4a`, or `ogg`
file up to 25 MB. Three sample calls are in `static/audio/`.

For anything beyond local use, run it under a real server:

```bash
gunicorn -w 1 -t 300 app:app
```

**One worker.** Each worker loads its own 4.5 GB of weights, so `-w 4` needs
roughly 18 GB of RAM. The long timeout matters too — a cold first request has to
download and load everything before it returns.

## API

```bash
curl -F "audio_file=@call.mp3" http://localhost:5000/api/process
```

Returns the full analysis as JSON — transcript, language, type, severity,
entities, summary, sentiment, emotion, dispatch, and `report_id`.

| Route | Purpose |
|---|---|
| `GET /` | Upload form |
| `POST /` | Upload and render results |
| `POST /api/process` | Same pipeline, JSON response |
| `GET /download_fir/<report_id>` | Fetch that request's PDF |
| `GET /audio/<filename>` | Serve an uploaded recording |
| `GET /healthz` | Liveness probe |

## Configuration

All of `config.py` reads from the environment:

| Variable | Default | |
|---|---|---|
| `WHISPER_MODEL_NAME` | `base` | `tiny` is ~3× faster, `small`/`medium` more accurate |
| `MAX_CONTENT_LENGTH` | 25 MB | Upload cap |
| `UPLOAD_FOLDER` | `static/uploads` | |
| `PROCESSED_FOLDER` | `processed` | PDFs and plots |
| `FLASK_DEBUG` | `0` | |
| `FORCE_CPU` | `1` | |

## Limitations

Things that are honestly not production-ready, stated plainly.

### The dispatch layer is a stub

`STATIONS` in `utils/audio_utils.py` is nine hardcoded fictional stations
("Central Hospital", "Precinct A") matched by keyword against the extracted
location string, with fixed base ETAs. There is no geocoding, no routing, no
real station registry, and the ETA is a constant, not a computation. It
demonstrates the shape of the integration point. It is not a dispatch system.

### Severity is a heuristic, not a calibrated model

The weights (0.4 / 0.2 / 0.2 / 0.2) are hand-chosen and have never been fitted
or validated against labelled call data. Zero-shot classification over
`high/medium/low` has no notion of what those words mean in an emergency-
dispatch context. Treat the output as a sortable signal, not a triage decision.

### Other

- **No authentication or rate limiting.** Every request costs seconds of CPU;
  `/api/process` is an open invitation to exhaust it.
- **No persistence.** Reports are written to disk and never cleaned up. There's
  no database and no retention policy — relevant, since call recordings are
  sensitive.
- **CPU-only by design**, so expect tens of seconds per call. `FORCE_CPU` exists
  but nothing in the pipeline currently reads it.
- **No tests.** `test.py` is a scratch script, not a suite.
- **Uploads are kept forever** under `static/uploads/` and served back publicly
  by filename.

## Recently fixed

Three bugs in the severity path, all of which silently produced wrong output
rather than failing:

- **The severity classifier was `microsoft/deberta-v3-base`**, which has no NLI
  head. The zero-shot pipeline attaches a randomly initialised classification
  head to such a model, so every severity score it ever produced was untrained
  noise. It now reuses the MNLI model, which also saves ~400 MB of RAM.
- **Severity read the classifier's confidence but discarded its label** —
  `scores[0]` is the top-ranked label's score whichever label that is, so a
  confident "low" scored identically to a confident "high". It now takes an
  expectation over the label distribution.
- **Entities were always empty.** `assess_severity(text, [])` was called with a
  hardcoded empty list before NER had run, so the entity term of the score was
  dead on every request. NER now runs first.

And one concurrency bug: the PDF was written to a single shared
`processed/fir_report.pdf` and `/download_fir` served that fixed path, so two
simultaneous callers overwrote each other and one could download the other's
report. Reports are now keyed by request ID.

## Licence

No licence yet — contact the owner before reuse.
