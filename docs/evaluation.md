# Evaluation

Two harnesses, measuring different things.

| | what it measures | run |
|---|---|---|
| `eval/run.py` | extraction accuracy against hand-labelled calls | `python -m eval.run --repeat 3` |
| `scripts/calibrate_severity.py` | whether the severity score measures severity | `python scripts/calibrate_severity.py` |
| `scripts/benchmark.py` | end-to-end latency on real audio | `python scripts/benchmark.py --runs 3` |

Everything below is measured on **fourteen** hand-labelled calls. That is a
regression guard, not a benchmark, and the confidence intervals are printed
next to every number for that reason.

## Extraction

Six checks per call — incident type, severity, weapons, location stated,
location content, schema cleanliness — so 84 points.

```
TOTAL 81-82/84 over 3 runs (mean 81.3)
```

**Report a range, not a number.** Three consecutive runs on identical code and
identical transcripts gave 82, 81 and 81, and the *failing items moved between
runs*. A hosted mixture-of-experts model is not deterministic across requests
even at temperature 0. A single run is a draw from that distribution; reading
82 against 81 as a regression is a mistake the harness now prevents by
printing the spread.

## Severity

The weights in `utils/severity.py` were hand-chosen and had never been
compared to anything. They now have been.

| predictor | in band | 95% CI | under-triage | over-triage |
|---|---|---|---|---|
| always `low` | 2/14 · 14% | 4–40% | 12 | 0 |
| always `medium` | 4/14 · 29% | 12–55% | 10 | 0 |
| always `high` | 6/14 · 43% | 21–67% | 6 | 2 |
| always `critical` | 10/14 · 71% | 45–88% | 0 | 4 |
| keyword signal groups | 12/14 · 86% | 60–96% | 1 | 1 |
| **hand-chosen weights** | **4/14 · 29%** | 12–55% | **10** | 0 |
| fitted weights, in-sample | 5/14 · 36% | 16–61% | 10 | 0 |
| fitted weights, leave-one-out | 3/14 · 21% | 8–48% | 10 | 0 |
| extraction model | 10/14 · 71% | 45–88% | 4 | 0 |
| **extraction model + keyword floor** | **13/14 · 93%** | 69–99% | **0** | 1 |

### What this says

**The weighted score does not measure severity.** It scores the same as
answering "medium" to every call. All ten of its failures are under-triage —
it said `medium` for an officer-involved shooting and `low` for a man found
cold and not breathing with CPR in progress.

**It never returns `high`.** The branch needs 0.7 and across fourteen real
calls the score never exceeded 0.64: the zero-shot classifier's distribution
is near-uniform, so the label term contributes about 0.25 of its 0.4 budget,
and sentiment and emotion are usually neutral and contribute nothing.

**Recalibrating the thresholds does not save it.** Fitted *on the test set* —
a cheating upper bound — the best three thresholds reach 10/14, the same as
answering "critical" to everything, and the thresholds it picks are degenerate.
The ordering is the reason: the score puts a break-in (0.640) above an
officer-involved shooting (0.435), and two teenagers fighting (0.460) above a
man not breathing (0.348).

**Fitting the weights makes it worse.** Four parameters over fourteen examples
overfits: 5/14 in-sample, 3/14 under leave-one-out. The hand-chosen weights
stay, because fitted ones are not better and pretending otherwise on n=14
would be the dishonest result.

**Grep beats it, 12/14 to 4/14.** That is the most interesting number here and
it is published rather than buried. The baseline is `utils.signals.detect_signals`
unchanged — the groups the pipeline already matches — not a word list written
afterwards. A list assembled after reading the labels would be fitted to them.

### What ships

`apply_severity_floor(severity, transcript)`, applied to whichever source
produced the severity, so the guarantee holds on every path:

```python
severity = apply_severity_floor(severity, transcript)
```

Weapons and life-threatening medical signals floor a call at `critical`, fire
and violence at `high`, vehicle at `medium`. The floor never lowers a
severity.

The combiner is a maximum rather than a blend because the two directions are
not equivalent: over-triage sends too much, under-triage sends too little to
someone who is dying. A maximum can only be wrong downward when *both* inputs
are wrong downward, and these two fail for unrelated reasons — one reads
words, the other reads meaning.

Measured: the model alone is 10/14 with four under-triages; with the floor,
13/14 with none.

### What this does not say

The labels are **one annotator's reading of fourteen transcripts**, with no
inter-annotator agreement, against the rubric committed in `eval/labels.json`.
This measures agreement with that rubric. It does not measure correctness
against a dispatch protocol, and the intervals above overlap almost completely
— treat the table as ordering evidence, not as measurement.

Severity was scored as a **ceiling** until 2026-09-22, which meant a model
answering `low` to every call passed 14/14 while one answering `critical`
passed 10/14. The check rewarded under-triage and penalised over-triage. It is
a band now, `severity_min..severity_max`.

## Latency

`scripts/benchmark.py` runs real audio end to end and compares against a saved
baseline, printing output differences as loudly as timing ones — a change that
got faster by disagreeing with itself is what it is for.

Local pipeline work is 1.3–3.2 s per call. Everything above that is the
extraction provider, which spread 1.8 s to 34 s on the same fourteen samples.
`--runs` above 1 measures the provider's rate limiter as much as anything
else, so watch the `local` line for changes to this repository.
