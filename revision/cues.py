"""The 22 search cues from Sec. III-A and helpers to find and mask them."""
import re

# Exactly as used for the thread search (Sec. III-A).
CUES = [
    "accidently", "discovered", "observed", "don't know", "seems", "found",
    "randomly", "looks like", "Why", "How", "What", "strange", "weird", "maybe",
    "I wonder", "I think", "I tried", "failure", "problem", "issue", "suddenly",
    "may have",
]

# Matching decisions (document these in the paper):
# - case-insensitive ("How" also matches "how")
# - whole words only ("issue" does NOT match "issues")
# - multi-word cues match across any whitespace
CASE_SENSITIVE = False


def normalize(text: str) -> str:
    """Unifies curly apostrophes so "don’t know" matches "don't know"."""
    return text.replace("\u2019", "'").replace("\u2018", "'")


def _compile(cue: str) -> re.Pattern:
    words = [re.escape(w) for w in cue.split()]
    flags = 0 if CASE_SENSITIVE else re.IGNORECASE
    return re.compile(r"\b" + r"\s+".join(words) + r"\b", flags)


PATTERNS = {cue: _compile(cue) for cue in CUES}


def find_cues(text: str) -> list[str]:
    """Return all cues occurring in the text."""
    text = normalize(text)
    return [cue for cue, pattern in PATTERNS.items() if pattern.search(text)]


def has_cue(text: str) -> bool:
    return bool(find_cues(text))


def mask_cues(text: str, mask_token: str) -> str:
    """Replace every word of every cue occurrence with mask_token.
    Used for the TF-IDF baseline.

    "I think" -> "<mask> <mask>". Use mask_token="" to delete cues instead
    """
    text = normalize(text)
    for pattern in PATTERNS.values():
        text = pattern.sub(lambda m: " ".join([mask_token] * len(m.group().split())), text)
    return text
