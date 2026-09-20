"""What runs, and when.

The pipeline used to be one fixed sequence: eight local models, always, in the
same order. That made sense when there was nothing else. A hosted extraction
model that scores 82/84 against the local path's 11/18, in a fifth of the
time, turns several of those stages into work whose output is discarded.

Deciding *not* to run a stage is a policy question, not a modelling one, so it
lives here -- importable without torch, and tested without loading anything.
"""
import os

# Below this the model has not really summarised anything -- an empty string,
# a fragment, or a refusal -- and BART is worth the minute it costs.
MIN_MODEL_SUMMARY_WORDS = 8

MODES = ("auto", "always", "never")


def mode():
    """`LOCAL_ANALYSIS`: auto | always | never. Anything else means auto."""
    chosen = os.environ.get("LOCAL_ANALYSIS", "auto").strip().lower()
    return chosen if chosen in MODES else "auto"


def local_analysis_wanted(record, chosen=None):
    """Should the BART classifiers run as a second opinion on this call?

    - ``auto`` (default) -- no, when the model returned a usable record.
      Running two BART-large passes to produce a worse version of an answer
      already in hand buys nothing but latency.
    - ``always`` -- yes, and the result page shows where the two disagree. A
      disagreement is a reason for a person to look, and that is worth about
      seventy seconds to some operators.
    - ``never`` -- no, ever. The classification is then whatever the model
      said, or ``unknown`` if there is no model.

    The floor still holds. With no model, a failed extraction, or a record
    missing the fields, the local path runs. It is the fallback now rather
    than the baseline, which is a real change in this project's premise.
    """
    chosen = chosen or mode()
    if chosen == "always":
        return True
    if chosen == "never":
        return False
    if not record:
        return True
    # A record naming neither a type nor a severity has not classified
    # anything, whatever its status field said.
    return not (record.get("incident_type") and record.get("severity"))


def model_summary(record):
    """The model's summary, if it produced one worth using.

    The gate is deliberately a word count and not `summary.is_informative`.
    That gate catches an *extractive* summariser drifting from its source, a
    failure mode a structured extraction does not have: the model returns one
    field of a validated record or it returns nothing. Running the lexical
    overlap check here would reject a good abstractive summary for the crime
    of using different words, then spend a minute of BART replacing it with a
    worse one.
    """
    text = ((record or {}).get("summary") or "").strip()
    return text if len(text.split()) >= MIN_MODEL_SUMMARY_WORDS else ""
