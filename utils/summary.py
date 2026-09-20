"""Summary length budgeting and quality gating, without the models.

Two problems, both seen on real calls.

**Fixed output lengths.** The summariser was asked for `max_length=120` tokens
regardless of input, so on a short transcript it warned

    Your max_length is set to 120, but your input_length is only 82

and padded to fill the budget. Padding is exactly the kind of fluent, empty
text the quality gate then rejected.

**A gate that assumed the emergency type.** `is_summary_informative` required a
word from `address, weapon, gun, shot, fire, injury, bleeding, ...`. A domestic
disturbance with no weapon, no vehicle and no injury contains none of them, so
every summary of such a call was rejected by construction and the user was shown
nothing at all.
"""
import re

STOPWORDS = frozenset("""
a an and are as at be been but by for from had has have he her his i if in is it
its me my not of on or our she that the their them there they this to was we
were what when where which who will with would you your ok okay yes no
""".split())


def length_budget(input_tokens, ratio=0.6, floor=25, ceiling=160):
    """Token budget for a summary of an input this long.

    A summary should be shorter than its source; asking for more invites
    padding. Returns (max_length, min_length) with min below max always.
    """
    if input_tokens <= 0:
        return floor, max(1, floor // 3)
    max_length = int(max(floor, min(ceiling, input_tokens * ratio)))
    min_length = max(1, min(int(max_length * 0.4), max(1, input_tokens // 3)))
    if min_length >= max_length:
        min_length = max(1, max_length - 1)
    return max_length, min_length


def content_words(text):
    return [w for w in re.findall(r"[a-z']+", (text or "").lower())
            if w not in STOPWORDS and len(w) > 2]


def is_informative(summary, source, min_words=12, max_ratio=0.9,
                   min_overlap=0.25):
    """Does this summary say something the source says, more briefly?

    Category-agnostic on purpose. Three checks:

    - long enough to be a summary rather than a fragment
    - actually shorter than the source
    - shares enough content words with the source to not be invented

    The last one is the useful part: an abstractive summariser that drifts
    produces fluent text with little lexical overlap, which is the failure worth
    catching. A whitelist of emergency nouns caught the wrong thing.
    """
    summary_words = (summary or "").split()
    if len(summary_words) < min_words:
        return False

    source_words = (source or "").split()
    if source_words and len(summary_words) > len(source_words) * max_ratio:
        return False

    summary_content = set(content_words(summary))
    source_content = set(content_words(source))
    if not summary_content or not source_content:
        return False

    overlap = len(summary_content & source_content) / len(summary_content)
    return overlap >= min_overlap
