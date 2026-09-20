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
