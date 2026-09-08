# Architecture

What the code does, and why it does it that way.

## Shape

```
app.py              Flask routes; upload validation, PDF download, audio serving
config.py           Environment-backed settings
utils/
  audio_utils.py    Model loaders, taxonomy, scoring, transcription, PDF
  plots.py          Plot rendering, free of ML imports so it stays testable
  retention.py      Age-based cleanup of stored audio and artefacts
tests/              Runs without the ML stack installed
templates/          Jinja
static/             CSS
storage/            Uploads and generated artefacts, outside the static mount
```

`utils/audio_utils.py:process_audio_file` is the entire pipeline. Everything
below happens inside a single HTTP request.

## The pipeline

| # | Stage | Notes |
|---|---|---|
| 1 | Load audio | 16 kHz mono, truncated to 120 s |
| 2 | Visualise | Waveform, MFCC, pitch |
| 3 | Transcribe | Whisper, language auto-detect |
| 4 | Translate | Only when detected language is not English |
| 5 | NER | spaCy, **before** severity; entity-label plot written here |
| 6 | Classify type | Zero-shot over four labels |
| 7 | Assess severity | Weighted: label distribution, sentiment, emotion, entity count |
| 8 | Summarise | BART chunked, T5 fallback, informativeness gate |
| 9 | Dispatch hint | Keyword match against a hardcoded station list |
| 10 | Render PDF | Keyed by `report_id` |

### Ordering that matters

**NER runs before severity.** Severity weights entity count. An earlier version
called `assess_severity(text, [])` with a hardcoded empty list before NER had
run, so that term of the score was dead on every request. The ordering is load-
bearing, not incidental.

**The entity plot is generated after NER, not with the other plots.**
`generate_visualizations` runs on raw audio at step 2, before entities exist, so
the entity distribution cannot be produced there. It has its own function, and
it returns a flag the template uses to decide whether to render the image at
all — a call in which spaCy recognises nothing is normal, and an empty figure is
worse than no figure.

**`report_id` is generated before the first artefact is written.** Every plot,
the intermediate WAV and the PDF are keyed by it, and each is served through a
route that validates the id against `[0-9a-f]{32}` before touching the
filesystem. Artefacts are per-request by construction rather than by
convention.

## Design notes

### One model, two classifiers

Both emergency-type classification and severity assessment run through the same
`facebook/bart-large-mnli` zero-shot pipeline.

Severity previously used `microsoft/deberta-v3-base`. **That checkpoint has no
NLI head.** HuggingFace's `zero-shot-classification` pipeline will happily
accept it and attach a *randomly initialised* classification head, which
produces confident, plausible, entirely meaningless labels — with no error and
no warning.

Reusing the MNLI pipeline is correct for the task and saves ~400 MB of RAM. The
general lesson: **a zero-shot pipeline is only as valid as the NLI head behind
it**, and nothing in the API tells you whether one exists.

### Severity is an expectation, not a top-1 score

```python
by_label = dict(zip(severity_result['labels'], severity_result['scores']))
severity_score += 0.4 * (by_label.get('high', 0.0) + 0.5 * by_label.get('medium', 0.0))
```

`scores[0]` is the confidence of whichever label ranked first. Reading it
directly scores a confident `low` identically to a confident `high` — it takes
the confidence and discards the class. Taking an expectation over the label
distribution is the fix.

The 0.4 / 0.2 / 0.2 / 0.2 weights are **hand-chosen and unvalidated.** They are
constants in the source, not learned parameters.

### Lazy loading

Eight models, ~4.5 GB. Each loader assigns to a module-level global and returns
early if it is already populated, so the first request pays the cost and the
rest do not.

This is also why the app runs one worker: the globals are per-process, so `-w 4`
means four independent 4.5 GB copies.

### Bounded input

Two caps, both to bound worst-case request time rather than for correctness:

- **25 MB upload** (`MAX_CONTENT_LENGTH`)
- **120 s of audio**, truncated after decode

Whisper is roughly linear in audio duration on CPU, so without the second cap a
single 40-minute recording would occupy the only worker for a very long time.

### Summarisation has a quality gate

BART-large-cnn over ~700-char chunks with 120-char overlap, then a final pass to
tighten. If `is_summary_informative` rejects the result, it falls back to T5; if
that is also rejected, the summary is dropped entirely rather than shown.

Showing an empty summary is better than showing a fluent one that contains no
information, because a fluent summary reads as a result.

### Per-request artefacts

`report_id` (`uuid4().hex`) keys the intermediate WAV and the output PDF.
`/download_fir/<report_id>` validates against `[0-9a-f]{32}` before touching the
filesystem — the path segment is user-controlled, so it needs the guard.

The same scheme covers the plots, which are written to
`<kind>_<report_id>.png` and served by `/plot/<report_id>/<kind>` against a
fixed set of kinds.

### Retention

`utils/retention.py` sweeps uploads and artefacts older than
`RETENTION_SECONDS`, on a timer and at startup. It takes an injectable clock so
the boundary is testable without sleeping, tolerates missing folders because it
runs before they may exist, and ignores files that vanish mid-sweep because two
workers sweeping at once is supported rather than guarded against.

### Testing without the models

`tests/conftest.py` stubs `utils.audio_utils`, so `app.py`, `config.py` and the
template are exercised as the real files while torch, whisper, spacy and
transformers are never imported. Plot and retention logic live in modules with
no ML imports for the same reason. The suite runs in about two seconds.
