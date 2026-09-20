"""Verify that quoted evidence actually appears in the transcript.

A model asked to justify a claim with a verbatim quote can still invent the
quote. That is mechanically checkable, and it is the difference between an
extraction you can audit and one you have to trust.

Matching is deliberately tolerant of punctuation, casing and whitespace, and
deliberately intolerant of paraphrase. ASR output has inconsistent punctuation,
so requiring an exact string match would reject honest quotes; allowing fuzzy
word overlap would accept invented ones. Normalised substring is the line.
"""
import re

_NON_WORD = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


def normalise(text):
    text = (text or "").lower()
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def quote_supported(quote, transcript, min_words=2):
    """Is this quote genuinely present in the transcript?

    Quotes shorter than `min_words` are rejected: a single word is not evidence,
    and "gun" appearing somewhere in a long transcript says nothing about
    whether a gun is present.
    """
    q = normalise(quote)
    if len(q.split()) < min_words:
        return False
    return q in normalise(transcript)


def verify_evidence(record, transcript, field="weapons"):
    """Drop members of `field` whose quote is not in the transcript.

    Returns (kept, dropped). Dropped entries are reported rather than discarded
    silently -- a model that repeatedly invents quotes is telling you something
    about the prompt.
    """
    entries = record.get(field) or []
    kept, dropped = [], []
    for entry in entries:
        if not isinstance(entry, dict):
            dropped.append({"item": str(entry), "reason": "no evidence supplied"})
            continue
        quote = entry.get("quote", "")
        if quote_supported(quote, transcript):
            kept.append(entry)
        else:
            dropped.append({**entry, "reason": "quote not found in transcript"})
    return kept, dropped
