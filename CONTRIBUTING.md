# Contributing

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
sudo apt install ffmpeg   # or: brew install ffmpeg
```

First run downloads ~4.5 GB of model weights into the HuggingFace cache. Budget
for it.

## Before you open a PR

Run the suite. It does not need the ML stack:

```bash
pip install -r requirements-dev.txt
pytest -q
```

For changes to the pipeline itself, also run the app against a recording of
your own and check the output by hand — no audio ships with the repository, by
design. Say in the PR description what you actually ran.

Say in the PR description what you actually ran. "Syntax-checked only" is an
acceptable answer; claiming otherwise is not.

## The thing to be careful about

Every bug found in this project so far has been **silent** — plausible output,
no error, no warning. A zero-shot pipeline pointed at a checkpoint with no NLI
head returns confident labels from a randomly initialised head. A classifier
score read without its label ranks a confident "low" like a confident "high". A
severity function called with an empty entity list still returns a severity.

None of those raised. None were visible in the UI. They were found by reading.

So: **prefer failing loudly over degrading quietly.** If a stage cannot do its
job, raise or record a warning that reaches the response — do not substitute a
default and return it as though it were a result.

Where a fix is behavioural, add a test that fails without it. Several tests in
`tests/` exist to pin exactly this kind of bug and are commented as such.

## Style

- Comments explain *why*, especially where a line encodes a decision that looks
  arbitrary — the 120 s truncation and the 25 MB cap both exist to bound
  latency, and both say so in the source.
- Keep the docs true. If a change alters behaviour the README describes,
  update the README in the same commit.

## Do not commit

Call recordings, transcripts, model checkpoints, or anything else derived from
real emergency calls. See [SECURITY.md](SECURITY.md#do-not-commit-call-data).
