"""Detect high-signal phrases in a transcript.

Split out of audio_utils so it can be tested without the ML stack, and because
the matching rule here turned out to matter more than any model in the pipeline.

**Word boundaries are the whole point.** This logic previously used
`substring in transcript`, with `'ex'` and `'car'` among the keywords:

- `'ex'` matches inside *excuse, exit, next, text, example, expecting,
  exactly, experience*. A dispatcher saying "excuse me" — which happens on
  most calls — triggered "Check for restraining order or prior incidents".
- `'car'` matches inside *cardiac, scared, card, carton, carpet*. A caller
  reporting a **cardiac** arrest triggered "Notify traffic control for scene
  safety".

Observed on a real sample call, not hypothesised.
"""
import re

# Each group: the phrases that trigger it, and the actions it adds.
SIGNAL_GROUPS = [
    ("violence", ["domestic", "ex-partner", "ex-boyfriend", "ex-girlfriend",
                  "ex-husband", "ex-wife", "restraining order", "stalking",
                  "stalker", "threatened", "threatening", "threaten", "violent"],
     ["Dispatch police to secure the scene",
      "Check for restraining order or prior incidents",
      "Advise caller to stay in a safe locked room"]),

    ("weapons", ["shot", "shots", "shoot", "shooting", "gun", "guns", "gunfire",
                 "firearm", "rifle", "pistol", "armed"],
     ["Issue officer safety advisory (possible firearm)",
      "Request additional police units"]),

    ("medical", ["not breathing", "unconscious", "severe bleeding", "cpr",
                 "cardiac", "seizure", "overdose", "choking"],
     ["Provide pre-arrival instructions (CPR/bleeding control)",
      "Dispatch ALS ambulance"]),

    ("fire", ["smoke", "flames", "burning", "gas leak", "explosion", "fire"],
     ["Shut off gas/electric if safe",
      "Keep bystanders clear and evacuate adjacent units"]),

    ("vehicle", ["vehicle", "truck", "car", "cars", "crash", "collision",
                 "accident", "rollover"],
     ["Notify traffic control for scene safety"]),
]


def _pattern(phrases):
    """Match whole words only. Multi-word phrases keep their internal spacing."""
    return re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(p) for p in phrases) + r")(?!\w)")


_COMPILED = [(name, _pattern(words), actions) for name, words, actions in SIGNAL_GROUPS]


def detect_signals(transcript):
    """Return {group: [matched phrases]} for whole-word matches only."""
    text = (transcript or "").lower()
    found = {}
    for name, pattern, _ in _COMPILED:
        hits = sorted(set(pattern.findall(text)))
        if hits:
            found[name] = hits
    return found


def augment_actions(transcript, base_response):
    """Add transcript-derived actions to a response, in place.

    Records what fired and why under `signals`, so a reader can see that an
    officer-safety advisory came from the word "shoot" rather than from a
    judgement. Keyword matching cannot tell a threat from a figure of speech --
    on the sample call, "did you want us to come over to shoot her?" was a
    dispatcher's sarcasm, and it still fires. Surfacing the evidence is the
    honest response to that; suppressing the advisory would be worse.
    """
    signals = detect_signals(transcript)
    suggestions = list(base_response.get("suggestions", []))
    for name, _, actions in _COMPILED:
        if name in signals:
            suggestions += actions

    seen = set()
    deduped = []
    for s in suggestions:
        if s not in seen:
            deduped.append(s)
            seen.add(s)

    base_response["suggestions"] = deduped
    base_response["signals"] = signals
    return base_response
