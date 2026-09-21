"""Repeatable evaluation of the extraction stage.

Deliberately runs on **saved transcripts**, not audio. Comparing local models
against each other does not need the ASR re-run, and separating the two means a
model comparison takes minutes instead of hours -- the reason the last one was
never done.

    python -m eval.run --transcripts eval/transcripts.json --model Qwen/Qwen2.5-1.5B-Instruct

Writes one JSON per configuration to eval/results/ so runs can be diffed.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the app's config for its side effect: it loads `.env`. Without this the
# harness sees none of the project's settings and reports "needs NVIDIA_API_KEY"
# on a machine where the key is configured correctly.
import config  # noqa: F401,E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEVERITY_ORDER = ["unknown", "low", "medium", "high", "critical"]

# An extracted address is a short span, not a paragraph of transcript.
MAX_LOCATION_CHARS = 80


def score(record, truth):
    """Compare one extraction against the labels. Returns per-field results.

    Severity is scored as a ceiling, not equality: labels record the *most*
    severe defensible answer, because reasonable people disagree between low
    and medium and nobody should disagree that two children fighting is not
    critical.
    """
    checks = {}

    got = record.get("incident_type")
    checks["incident_type"] = {"expected": truth["incident_type"], "got": got,
                               "pass": got == truth["incident_type"]}

    got = record.get("severity") or "unknown"
    ceiling = truth["severity_max"]
    ok = (SEVERITY_ORDER.index(got) <= SEVERITY_ORDER.index(ceiling)
          if got in SEVERITY_ORDER and ceiling in SEVERITY_ORDER else False)
    checks["severity"] = {"expected": f"<= {ceiling}", "got": got, "pass": ok}

    weapons = record.get("weapons") or []
    names = [w.get("item") if isinstance(w, dict) else w for w in weapons]
    checks["weapons"] = {"expected": truth["weapon_present"],
                         "got": names, "pass": bool(weapons) == truth["weapon_present"]}

    location = record.get("location")
    if location is not None and not isinstance(location, str):
        location = str(location)          # never crash the harness on a shape
    stated = bool(location)
    ok = stated == truth["location_stated"]
    if ok and stated and truth.get("location_contains"):
        # Substring alone is too lenient: a model that dumps 200 characters of
        # transcript into `location` passes a `contains` check while extracting
        # nothing. An address is short; require that too.
        ok = (truth["location_contains"].lower() in location.lower()
              and len(location) <= MAX_LOCATION_CHARS)
    checks["location"] = {"expected": truth.get("location_contains") if truth["location_stated"] else None,
                          "got": location, "pass": ok}

    checks["uncertainties_present"] = {
        "expected": True, "got": len(record.get("uncertainties") or []),
        "pass": bool(record.get("uncertainties"))}

    checks["schema_clean"] = {
        "expected": "no rejected fields", "got": record.get("_rejected"),
        "pass": not record.get("_rejected")}

    return checks


def model_env(backend):
    """Which environment variable names the model for this backend."""
    return "LOCAL_LLM_MODEL" if backend == "local" else "LLM_MODEL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", default=os.path.join(HERE, "transcripts.json"))
    ap.add_argument("--labels", default=os.path.join(HERE, "labels.json"))
    ap.add_argument("--model", help="model id, for whichever backend is chosen")
    ap.add_argument("--device", help="LOCAL_LLM_DEVICE override; local only")
    ap.add_argument("--backend", default=os.environ.get("LLM_BACKEND", "local"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    os.environ["LLM_BACKEND"] = args.backend
    if args.device:
        os.environ["LOCAL_LLM_DEVICE"] = args.device
    if args.model:
        # The two backends read different variables, and setting the wrong one
        # is silent: `--model openai/gpt-oss-120b --backend groq` ran Groq's
        # *default* model and printed its name, so a bake-off between two
        # hosted models would have compared one model with itself.
        os.environ[model_env(args.backend)] = args.model

    from utils.llm import extract_incident, get_backend

    transcripts = json.load(open(args.transcripts))
    labels = json.load(open(args.labels))

    t0 = time.perf_counter()
    backend = get_backend()
    load_s = round(time.perf_counter() - t0, 1)
    if backend is None:
        from utils import llm as llm_mod
        print(f"FAILED to load backend: {llm_mod._LAST_ERROR}")
        return 1
    label = backend.describe()
    print(f"backend={backend.name} model={label} load={load_s}s")
    if args.model and args.model not in label:
        print(f"WARNING: asked for {args.model!r} but the backend reports "
              f"{label!r} -- the results below are not for the model you named.")
    print()

    rows, passed, total = [], 0, 0
    for sample, entry in sorted(transcripts.items()):
        truth = labels.get(sample)
        if not truth:
            continue
        t0 = time.perf_counter()
        record, meta = extract_incident(
            entry["text"], truncated=entry.get("truncated", False),
            source_seconds=entry.get("source_seconds"),
            analysed_seconds=entry.get("analysed_seconds"))
        elapsed = round(time.perf_counter() - t0, 1)
        if record is None:
            print(f"{sample:14} FAILED {meta.get('reason','')[:80]}")
            rows.append({"sample": sample, "error": meta})
            continue

        checks = score(record, truth)
        ok = sum(1 for c in checks.values() if c["pass"])
        passed += ok
        total += len(checks)
        print(f"{sample:14} {ok}/{len(checks)}  {elapsed:6.1f}s")
        for name, c in checks.items():
            if not c["pass"]:
                print(f"    FAIL {name:22} expected {c['expected']!r}  got {c['got']!r}")
        rows.append({"sample": sample, "seconds": elapsed,
                     "checks": checks, "record": record, "meta": meta})

    print(f"\nTOTAL {passed}/{total}")
    os.makedirs(args.out, exist_ok=True)
    slug = label.replace("/", "_").replace(" ", "").replace("(", "").replace(")", "")
    path = os.path.join(args.out, f"{backend.name}__{slug}.json")
    json.dump({"backend": backend.name, "model": label, "load_seconds": load_s,
               "passed": passed, "total": total, "rows": rows},
              open(path, "w"), indent=2, default=str)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
