"""Severity scoring, as a pure function over already-extracted signals.

Lives outside audio_utils so it can be tested without loading any model. The
weights are still hand-chosen and unvalidated -- what changed is which inputs
are allowed to move the score.

Two of the original four terms measured something other than severity, observed
on a real call:

- **Entity count.** `min(len(entities) * 0.1, 0.2)` saturated at two entities.
  Of the 16 spaCy found on that call, 12 were bare numbers -- "911", ages, a
  partial phone number. Every emergency call contains those, so the term was a
  constant +0.2 offset rather than a signal.
- **Emotion below chance.** The detector returned `surprise` at 0.261 across
  seven classes, where chance is 0.143, and that was multiplied by 0.2 and
  added with no threshold.
"""

WEIGHT_LABEL = 0.4
WEIGHT_SENTIMENT = 0.2
WEIGHT_EMOTION = 0.2
WEIGHT_ENTITIES = 0.2

HIGH_THRESHOLD = 0.7
MEDIUM_THRESHOLD = 0.4

DISTRESS_EMOTIONS = ("fear", "anger", "sadness", "disgust")

# Entity types that say something about an emergency. CARDINAL and DATE are
# excluded on purpose: they capture "911", ages and phone digits, which is why
# the old count-everything term was saturated on every call.
SEVERITY_ENTITY_LABELS = frozenset({
    "PERSON", "GPE", "LOC", "FAC", "ORG", "QUANTITY", "TIME", "EVENT", "NORP",
})


def relevant_entities(entities):
    """Entities that plausibly bear on severity, by label."""
    return [e for e in entities or []
            if (e.get("label") if isinstance(e, dict) else None)
            in SEVERITY_ENTITY_LABELS]


def severity_score(label_distribution, sentiment, emotion, entities,
                   emotion_min_confidence=0.5):
    """Combine the signals into a 0-1 score. Returns (score, contributions).

    `contributions` is returned so a reader -- or a test -- can see which term
    moved the result, rather than inferring it from one number.
    """
    dist = label_distribution or {}
    parts = {}

    # Expectation over the label distribution, not the top-1 confidence: reading
    # scores[0] scored a confident "low" exactly like a confident "high".
    parts["label"] = WEIGHT_LABEL * (dist.get("high", 0.0)
                                     + 0.5 * dist.get("medium", 0.0))

    sentiment = sentiment or {}
    parts["sentiment"] = (WEIGHT_SENTIMENT * float(sentiment.get("score", 0.0))
                          if sentiment.get("label") == "NEG" else 0.0)

    emotion = emotion or {}
    emotion_score = float(emotion.get("score", 0.0))
    parts["emotion"] = (
        WEIGHT_EMOTION * emotion_score
        if emotion.get("label") in DISTRESS_EMOTIONS
        and emotion_score >= emotion_min_confidence
        else 0.0)

    relevant = relevant_entities(entities)
    parts["entities"] = min(len(relevant) * 0.05, WEIGHT_ENTITIES)

    return sum(parts.values()), parts


def severity_label(score):
    if score > HIGH_THRESHOLD:
        return "high"
    if score > MEDIUM_THRESHOLD:
        return "medium"
    return "low"


# ---- The floor ---------------------------------------------------------------
# Measured on the fourteen labelled calls (FEAT-014, scripts/calibrate_severity.py):
#
#     signal groups alone            12/14   1 under-triage
#     extraction model alone         10/14   4 under-triage
#     extraction model + this floor  13/14   0 under-triage
#     weighted score above            4/14  10 under-triage
#
# The weighted score is kept because it is what runs when there is no model
# and no match, but it is not trusted alone. It scored the same as answering
# "medium" to everything, it has never once returned "high" -- its arithmetic
# tops out at 0.64 against a 0.7 threshold, so that branch is unreachable --
# and its ordering puts a break-in above an officer-involved shooting.
#
# The asymmetry is the argument for a floor rather than a blend. Over-triage
# sends too much; under-triage sends too little to someone who is dying. A
# maximum can only be wrong downward when *both* inputs are wrong downward,
# and these two fail for unrelated reasons: one reads words, the other reads
# meaning.

SEVERITY_ORDER = ("unknown", "low", "medium", "high", "critical")

# Which band each signal group implies on its own. Weapons and
# life-threatening medical are the two categories that kill people.
SIGNAL_BAND = {
    "weapons": "critical",
    "medical": "critical",
    "fire": "high",
    "violence": "high",
    "vehicle": "medium",
}


def severity_from_signals(transcript, with_evidence=False):
    """The band the transcript's own words imply, or None if they imply nothing.

    Deliberately the *shipped* signal groups from `utils.signals`, not a word
    list written for this purpose. A list assembled after reading the labels
    would be fitted to them, and a baseline built that way measures its author
    rather than the method.

    `with_evidence` also returns the words that fired. Keyword matching cannot
    tell a threat from a figure of speech -- on one labelled call a
    dispatcher's sarcastic "shoot her" floors a squabble between two
    teenagers at critical, which is the single over-triage in the measured
    set. Suppressing the floor would cost four under-triages to save that
    one; showing the words instead lets a reader discount it in a second.
    """
    from utils.signals import detect_signals

    found = detect_signals(transcript)
    scored = {g: SIGNAL_BAND[g] for g in found if g in SIGNAL_BAND}
    band = max(scored.values(), key=SEVERITY_ORDER.index) if scored else None
    if not with_evidence:
        return band
    evidence = sorted({w for g, b in scored.items() if b == band
                       for w in found[g]})
    return band, evidence


def apply_severity_floor(severity, transcript):
    """Raise a severity to the floor its transcript implies. Never lowers it.

    Applied to whichever source produced the severity -- the extraction model
    or the weighted score -- so the guarantee holds on every path rather than
    on whichever one happened to run.
    """
    current = severity if severity in SEVERITY_ORDER else "unknown"
    floor = severity_from_signals(transcript)
    if floor is None:
        return current
    return max(current, floor, key=SEVERITY_ORDER.index)
