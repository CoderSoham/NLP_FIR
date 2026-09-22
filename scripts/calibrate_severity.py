#!/usr/bin/env python3
"""Fit and report the severity score against hand-labelled calls.

FEAT-014. The weights in `utils/severity.py` were hand-chosen and had never
been compared to anything. This answers three questions in the order that
matters:

1. **Does the score beat a constant?** Always answering "critical" is a real
   baseline and a hard one on this set. A scorer that cannot beat it is not
   measuring severity, it is decorating a guess.
2. **Does it beat grep?** Keyword matching on the words already hardcoded in
   `utils/signals.py`. If eight models cannot beat `grep -c shot`, that is the
   most interesting result in the repository and it gets published.
3. **Do fitted weights beat the hand-chosen ones?** Four parameters over
   fourteen examples, so leave-one-out cross-validation and nothing heavier.
   Anything more would overfit and lie about what fourteen labels support.

Two stages, deliberately separate:

    python scripts/calibrate_severity.py --extract   # runs the models once
    python scripts/calibrate_severity.py             # fits and reports

The first needs the ML stack and takes a couple of minutes. The second reads
the cached features, runs in milliseconds, and is what you re-run after
changing a weight -- so the fit is reproducible by someone without a GPU.

> The labels are one annotator's reading of fourteen transcripts. This
> measures **agreement with the rubric in eval/labels.json**, not correctness
> against dispatch protocol. With n=14 the confidence interval is enormous and
> the report prints it for that reason.
"""
import argparse
import itertools
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.severity import (WEIGHT_EMOTION, WEIGHT_ENTITIES, WEIGHT_LABEL,
                            WEIGHT_SENTIMENT, relevant_entities,
                            severity_label, severity_score)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVAL = os.path.join(ROOT, "eval")
FEATURES = os.path.join(EVAL, "severity_features.json")

ORDER = ["unknown", "low", "medium", "high", "critical"]

# The baseline uses `utils.signals.detect_signals` unchanged -- the exact
# groups the pipeline already matches on, not a word list written afterwards.
# That distinction is the whole value of the number: a list assembled after
# reading the labels would be fitted to them, and a 100% baseline built that
# way measures the author, not the method.
#
# The group-to-band mapping is the one judgement here, and it is made on
# principle rather than by trying combinations: weapons and life-threatening
# medical are the two categories that kill people.
SIGNAL_BAND = {
    "weapons": "critical",
    "medical": "critical",
    "fire": "high",
    "violence": "high",
    "vehicle": "medium",
}


def in_band(predicted, truth):
    """Is this prediction inside the labelled band?"""
    try:
        lo = ORDER.index(truth.get("severity_min", "unknown"))
        hi = ORDER.index(truth["severity_max"])
        return lo <= ORDER.index(predicted) <= hi
    except ValueError:
        return False


def distance(predicted, truth):
    """Bands away from the label. Negative means under-triage.

    Reported separately from accuracy because the two directions are not
    equivalent: over-triage sends too much, under-triage sends too little to
    someone who is dying.
    """
    try:
        got = ORDER.index(predicted)
        lo = ORDER.index(truth.get("severity_min", "unknown"))
        hi = ORDER.index(truth["severity_max"])
    except ValueError:
        return 0
    if got < lo:
        return got - lo
    if got > hi:
        return got - hi
    return 0


# ---- predictors --------------------------------------------------------------

def constant(level):
    return lambda _features: level


def keyword_predictor(features):
    """Highest band any shipped signal group implies. Whole-word, per ISSUE-025."""
    from utils.signals import detect_signals

    found = detect_signals(features.get("text") or "")
    bands = [SIGNAL_BAND[g] for g in found if g in SIGNAL_BAND]
    return max(bands, key=ORDER.index) if bands else "low"


def weighted_predictor(weights):
    def predict(features):
        score, _parts = severity_score(
            features.get("distribution"), features.get("sentiment"),
            features.get("emotion"), features.get("entities"))
        # severity_score bakes the module constants in, so rescale each term
        # by the ratio of the candidate weight to the shipped one.
        score, parts = severity_score(
            features.get("distribution"), features.get("sentiment"),
            features.get("emotion"), features.get("entities"))
        shipped = {"label": WEIGHT_LABEL, "sentiment": WEIGHT_SENTIMENT,
                   "emotion": WEIGHT_EMOTION, "entities": WEIGHT_ENTITIES}
        total = sum(parts[k] * (weights[k] / shipped[k]) if shipped[k] else 0.0
                    for k in parts)
        return severity_label(total)
    return predict


def llm_predictor(features):
    return features.get("llm_severity") or "unknown"


def floor_with(predict, floor=None):
    """Take whichever of two predictors says the higher band.

    The asymmetry is the argument. Over-triage sends too much; under-triage
    sends too little to someone who is dying. A max combiner can only be
    wrong downward when *both* inputs are wrong downward, and the two here
    fail for unrelated reasons -- one reads words, the other reads meaning.
    """
    floor = floor or keyword_predictor

    def predict_both(features):
        return max(predict(features), floor(features), key=ORDER.index)
    return predict_both


# ---- evaluation --------------------------------------------------------------

def evaluate(predict, samples, labels):
    hits, rows = 0, []
    for name in samples:
        truth = labels[name]
        got = predict(samples[name])
        ok = in_band(got, truth)
        hits += ok
        rows.append((name, got, truth.get("severity_min"),
                     truth["severity_max"], ok, distance(got, truth)))
    return hits, rows


def wilson(hits, n, z=1.96):
    """Wilson interval. With n=14 it is wide, which is the point of showing it."""
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def report(name, hits, rows):
    n = len(rows)
    lo, hi = wilson(hits, n)
    under = sum(1 for r in rows if r[5] < 0)
    over = sum(1 for r in rows if r[5] > 0)
    print(f"{name:26} {hits:2}/{n}  {hits / n:5.0%}  "
          f"[{lo:.0%}-{hi:.0%}]   under {under}  over {over}")
    return {"name": name, "hits": hits, "n": n, "under": under, "over": over,
            "ci": [round(lo, 3), round(hi, 3)]}


def fit_weights(samples, labels, step=0.1):
    """Grid search with leave-one-out cross-validation.

    Four parameters summing to 1.0 on a 0.1 grid is 286 combinations. Small
    enough to search exhaustively, which beats an optimiser nobody can audit.
    """
    grid = [w for w in itertools.product(
        [round(i * step, 2) for i in range(int(1 / step) + 1)], repeat=4)
        if abs(sum(w) - 1.0) < 1e-9]
    names = list(samples)

    def score_weights(w, subset):
        keys = ("label", "sentiment", "emotion", "entities")
        predict = weighted_predictor(dict(zip(keys, w)))
        return sum(in_band(predict(samples[k]), labels[k]) for k in subset)

    # Leave-one-out: fit on thirteen, predict the fourteenth. With n=14 this
    # is the only honest way to report a fitted number -- fitting and scoring
    # on the same fourteen examples would report the grid's best overfit.
    loo_hits = 0
    for held_out in names:
        rest = [k for k in names if k != held_out]
        best = max(grid, key=lambda w: score_weights(w, rest))
        keys = ("label", "sentiment", "emotion", "entities")
        loo_hits += in_band(
            weighted_predictor(dict(zip(keys, best)))(samples[held_out]),
            labels[held_out])

    best_overall = max(grid, key=lambda w: score_weights(w, names))
    keys = ("label", "sentiment", "emotion", "entities")
    return dict(zip(keys, best_overall)), loo_hits, len(grid)


# ---- feature extraction ------------------------------------------------------

def extract(transcripts):
    """Run the models once and cache what the scorer needs."""
    from utils.audio_utils import (load_emotion_detector, load_ner_model,
                                   load_sentiment_analyzer,
                                   load_severity_classifier)

    ner = load_ner_model()
    out = {}
    for name, entry in sorted(transcripts.items()):
        text = entry["text"]
        result = load_severity_classifier()(
            text[:500], candidate_labels=["high", "medium", "low"])
        try:
            sentiment = load_sentiment_analyzer()(text[:128])[0]
        except Exception as exc:                        # noqa: BLE001
            sentiment = {"label": "NEU", "score": 0.5, "_error": str(exc)[:80]}
        try:
            emotion = load_emotion_detector()(text[:128])[0]
        except Exception as exc:                        # noqa: BLE001
            emotion = {"label": "neutral", "score": 0.5, "_error": str(exc)[:80]}
        doc = ner(text)
        # No transcript text. It already lives in transcripts.json, and a
        # second tracked copy is a second place to redact -- this repository
        # has purged call transcripts from its history once already.
        out[name] = {
            "distribution": dict(zip(result["labels"], result["scores"])),
            "sentiment": sentiment,
            "emotion": emotion,
            "entities": [{"text": e.text, "label": e.label_} for e in doc.ents],
        }
        print(f"  {name:14} {out[name]['distribution']}")
    return out


def attach_llm(features, results_dir):
    """Fold in what the extraction model said, from the newest eval result."""
    newest, newest_time = None, 0
    for entry in os.scandir(results_dir) if os.path.isdir(results_dir) else []:
        if entry.name.endswith(".json") and entry.stat().st_mtime > newest_time:
            newest, newest_time = entry.path, entry.stat().st_mtime
    if not newest:
        return None
    data = json.load(open(newest))
    for row in data.get("rows", []):
        record = row.get("record") or {}
        if row.get("sample") in features:
            features[row["sample"]]["llm_severity"] = record.get("severity")
    return data.get("model")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", action="store_true",
                    help="run the models and cache the features")
    ap.add_argument("--features", default=FEATURES)
    ap.add_argument("--out", default=os.path.join(EVAL, "severity_calibration.json"))
    args = ap.parse_args(argv)

    transcripts = json.load(open(os.path.join(EVAL, "transcripts.json")))
    labels = {k: v for k, v in json.load(open(os.path.join(EVAL, "labels.json"))).items()
              if not k.startswith("_")}

    if args.extract:
        features = extract(transcripts)
        json.dump(features, open(args.features, "w"), indent=2)
        print(f"wrote {args.features}")
        return 0

    if not os.path.exists(args.features):
        raise SystemExit(f"no features at {args.features}; run --extract first")
    features = json.load(open(args.features))
    samples = {k: dict(v, text=transcripts.get(k, {}).get("text", ""))
               for k, v in features.items() if k in labels}
    llm_model = attach_llm(samples, os.path.join(EVAL, "results"))

    print(f"{len(samples)} labelled calls. Severity is scored as a band "
          f"(severity_min..severity_max).\n")
    print(f"{'predictor':26} {'score':>6} {'':5}  {'95% CI':^11}   direction")
    print("-" * 76)

    summary = []
    for level in ("low", "medium", "high", "critical"):
        hits, rows = evaluate(constant(level), samples, labels)
        summary.append(report(f"always {level}", hits, rows))

    hits, rows = evaluate(keyword_predictor, samples, labels)
    summary.append(report("keyword baseline", hits, rows))

    shipped = {"label": WEIGHT_LABEL, "sentiment": WEIGHT_SENTIMENT,
               "emotion": WEIGHT_EMOTION, "entities": WEIGHT_ENTITIES}
    hits, hand_rows = evaluate(weighted_predictor(shipped), samples, labels)
    summary.append(report("hand-chosen weights", hits, hand_rows))

    fitted, loo_hits, grid_size = fit_weights(samples, labels)
    fit_hits, _rows = evaluate(weighted_predictor(fitted), samples, labels)
    summary.append(report("fitted (in-sample)", fit_hits, hand_rows))
    summary.append(report("fitted (leave-one-out)", loo_hits, hand_rows))

    hits, rows = evaluate(floor_with(weighted_predictor(shipped)), samples, labels)
    summary.append(report("weights, keyword floor", hits, rows))

    if any(s.get("llm_severity") for s in samples.values()):
        hits, llm_rows = evaluate(llm_predictor, samples, labels)
        summary.append(report("extraction model", hits, llm_rows))
        hits, rows = evaluate(floor_with(llm_predictor), samples, labels)
        summary.append(report("model, keyword floor", hits, rows))

    print("-" * 76)
    print(f"hand-chosen weights: {shipped}")
    print(f"fitted weights:      {fitted}   ({grid_size} combinations searched)")
    if llm_model:
        print(f"extraction model:    {llm_model}")

    print("\nPer-call, hand-chosen weights:")
    for name, got, lo, hi, ok, dist in hand_rows:
        mark = "ok  " if ok else ("UNDER" if dist < 0 else "over ")
        print(f"  {name:14} said {got:9} label {lo}..{hi:9} {mark}")

    json.dump({"n": len(samples), "shipped": shipped, "fitted": fitted,
               "results": summary}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    print("\nn=14. The intervals above overlap almost completely, so treat "
          "these as\nordering evidence, not as measurements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
