#!/usr/bin/env python3
"""Time the whole pipeline across sample calls, and check it still works.

`eval/run.py` scores the extraction model on fixed transcripts. That is the
accuracy benchmark and it deliberately touches nothing else. This is the other
half: it runs real audio through `process_audio_file` end to end, so it
measures the stages the eval never sees -- decode, cleaning, plotting, NER,
report writing -- and catches a change that makes the pipeline fast by making
it wrong.

    python scripts/benchmark.py                     # every sample in static/audio
    python scripts/benchmark.py --runs 3            # median of three
    python scripts/benchmark.py --baseline old.json # compare against a saved run

Transcription is cached by content hash, so a second run measures the
pipeline rather than the ASR provider. Pass --no-cache to include it.

**`--runs` above 1 measures the provider's rate limiter.** Nine calls in a
minute against a free tier earns 429s, and the backoff shows up as every
sample taking a suspiciously similar thirty-odd seconds. The `local` column
is the number to watch for a change to this repository; the wait on the model
is not something this repository controls.
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402  - loads .env

# A run that got faster by dropping a field is not a faster run. These are the
# outputs a reader acts on, so the benchmark fails if any of them vanishes.
REQUIRED = ("transcription", "emergency_type", "severity", "report_id",
            "fir_pdf", "summary")


def run_once(path, output_folder):
    from utils.audio_utils import process_audio_file

    stages = {}
    marks = []

    def progress(stage, **_partial):
        now = time.perf_counter()
        if marks:
            stages[marks[-1][0]] = round(now - marks[-1][1], 2)
        if stage is not None:
            marks.append((stage, now))

    started = time.perf_counter()
    data = process_audio_file(path, output_folder, progress=progress)
    total = time.perf_counter() - started
    if marks:
        stages[marks[-1][0]] = round(time.perf_counter() - marks[-1][1], 2)

    missing = [f for f in REQUIRED if not data.get(f)]
    record = data.get("llm") or {}
    return {
        "seconds": round(total, 2),
        "stages": stages,
        "missing_fields": missing,
        "warnings": list(data.get("warnings") or []),
        "analysed_s": data.get("analysed_duration_s"),
        "truncated": bool(data.get("truncated")),
        "emergency_type": data.get("emergency_type"),
        "severity": data.get("severity"),
        "location": record.get("location"),
        "weapons": sorted(w.get("item") if isinstance(w, dict) else str(w)
                          for w in (record.get("weapons") or [])),
        "llm_seconds": (data.get("llm_meta") or {}).get("seconds"),
        "llm_waited": (data.get("llm_meta") or {}).get("waited_seconds"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", default="static/audio", help="directory of samples")
    ap.add_argument("--runs", type=int, default=1, help="repeats per sample")
    ap.add_argument("--out", default=None, help="write results as JSON")
    ap.add_argument("--baseline", default=None, help="compare against a saved run")
    ap.add_argument("--no-cache", action="store_true",
                    help="re-transcribe, measuring the ASR provider too")
    args = ap.parse_args(argv)

    if args.no_cache:
        os.environ["TRANSCRIPT_CACHE"] = ""

    samples = sorted(glob.glob(os.path.join(args.audio, "*.mp3"))
                     + glob.glob(os.path.join(args.audio, "*.wav")))
    if not samples:
        raise SystemExit(f"no audio in {args.audio}")

    results, failed = {}, []
    for path in samples:
        name = os.path.basename(path)
        runs = []
        for _ in range(args.runs):
            try:
                runs.append(run_once(path, config.PROCESSED_FOLDER))
            except Exception as exc:                    # noqa: BLE001
                failed.append((name, f"{type(exc).__name__}: {exc}"))
                break
        if not runs:
            continue
        # The median *run*, kept whole. Taking the fastest run's stage
        # breakdown and pairing it with the median total produced a record
        # whose stages did not add up to its own time.
        runs.sort(key=lambda r: r["seconds"])
        best = runs[len(runs) // 2]
        best["runs"] = [r["seconds"] for r in runs]
        results[name] = best
        flag = ""
        if best["missing_fields"]:
            flag = "  MISSING " + ",".join(best["missing_fields"])
        elif best["warnings"]:
            flag = "  WARN " + ";".join(best["warnings"])[:40]
        print(f"{name:16} {best['seconds']:6.2f}s  "
              f"{best['analysed_s'] or 0:7.1f}s audio  "
              f"{best['emergency_type'] or '?':9} {best['severity'] or '?':8}{flag}")

    if not results:
        raise SystemExit("every sample failed")

    times = [r["seconds"] for r in results.values()]
    audio = sum(r["analysed_s"] or 0 for r in results.values())
    print(f"\n{len(results)} samples, {audio:.0f}s of audio")
    # Local time is what a change here moves. Separating it from the wait on
    # the model is the difference between "the pipeline got slower" and "the
    # provider was busy", and those need different responses.
    local = [r["seconds"] - (r["llm_waited"] or 0) for r in results.values()]
    print(f"total {sum(times):.1f}s   median {statistics.median(times):.2f}s   "
          f"max {max(times):.2f}s   realtime factor {audio / sum(times):.0f}x")
    print(f"local pipeline only: median {statistics.median(local):.2f}s   "
          f"max {max(local):.2f}s   "
          f"({audio / max(sum(local), 0.01):.0f}x realtime)")

    for name, why in failed:
        print(f"FAILED {name}: {why}", file=sys.stderr)

    if args.baseline:
        compare(json.load(open(args.baseline)), results)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.out}")

    # Non-zero when a sample failed or lost a field, so this can gate a change.
    return 1 if failed or any(r["missing_fields"] for r in results.values()) else 0


def compare(baseline, now):
    """Timing and output differences against a saved run.

    Output differences are printed as loudly as timing ones. A change that
    made the pipeline faster by making it disagree with itself is the failure
    this whole script exists to catch.
    """
    print("\nagainst baseline:")
    regressed = False
    for name, after in sorted(now.items()):
        before = baseline.get(name)
        if not before:
            print(f"  {name:16} new")
            continue
        delta = after["seconds"] - before["seconds"]
        arrow = "faster" if delta < 0 else "SLOWER"
        print(f"  {name:16} {before['seconds']:6.2f}s -> {after['seconds']:6.2f}s  "
              f"{arrow} {abs(delta):.2f}s")
        for field in ("emergency_type", "severity", "location", "weapons"):
            if before.get(field) != after.get(field):
                regressed = True
                print(f"      {field}: {before.get(field)!r} -> {after.get(field)!r}")
    missing = set(baseline) - set(now)
    if missing:
        print(f"  gone from this run: {', '.join(sorted(missing))}")
    if regressed:
        print("  ^ outputs changed. Speed is not the only thing being measured.")


if __name__ == "__main__":
    raise SystemExit(main())
