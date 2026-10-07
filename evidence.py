"""Deterministic text utilities for relevance checks and evidence matching (Hebrew + English).

Used to (1) decide whether a government dataset is related to the research topic before its
contents are used, and (2) confirm that an excerpt the model quotes actually appears in content
that was retrieved during the run.
"""

import re
import sys
import unicodedata

HEBREW_FINALS = str.maketrans("ךםןףץ", "כמנפצ")
HEBREW_PREFIX_LETTERS = set("והבלמשכ")
# Common inflection suffixes, in normalized form (final letters unified: ם->מ, ן->נ).
HEBREW_SUFFIXES = {"", "ימ", "ות", "י", "ה", "ת", "יה", "יו", "ית", "נו", "כמ", "המ", "ני"}
STOPWORDS = {
    # English
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "not", "all", "any", "per",
    "into", "about", "under", "over", "data", "dataset", "datasets", "list", "table", "file",
    "government", "official", "israel", "israeli", "ministry", "information", "report",
    # Hebrew
    "של", "את", "על", "עם", "או", "גם", "כל", "לא", "אם", "כי", "זה", "זו", "אשר", "בין", "לפי",
    "עבור", "אל", "יש", "אין", "הוא", "היא", "נתונים", "מאגר", "קובץ", "טבלה", "רשימה", "רשימת",
    "משרד", "ממשלה", "ממשלתי", "ישראל", "מידע", "דוח",
}
_TOKEN_RE = re.compile(r"[\w֐-׿]+", re.UNICODE)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏‪-‮⁦-⁩]")


def strip_controls(text: str) -> str:
    """Remove control and bidi-override characters from untrusted content (keeps \\n and \\t)."""
    return _CONTROL_RE.sub("", text or "")


# Every combining character (niqqud, accents): removing them with one regex is equivalent to filtering
# unicodedata.combining() per character, and much faster on large dataset snapshots.
def _combining_class() -> str:
    ranges: list[list[int]] = []
    for c in range(sys.maxunicode + 1):
        if unicodedata.combining(chr(c)):
            if ranges and ranges[-1][1] == c - 1:
                ranges[-1][1] = c
            else:
                ranges.append([c, c])
    return "".join(re.escape(chr(a)) if a == b else f"{re.escape(chr(a))}-{re.escape(chr(b))}" for a, b in ranges)


_COMBINING_RE = re.compile(f"[{_combining_class()}]")
_COMBINING_SET = frozenset(c for c in map(chr, range(sys.maxunicode + 1)) if unicodedata.combining(c))
_FINALS = (("ך", "כ"), ("ם", "מ"), ("ן", "נ"), ("ף", "פ"), ("ץ", "צ"))


def normalize_text(text: str) -> str:
    """Lowercase, drop Hebrew niqqud, unify final letters, replace punctuation with spaces."""
    text = unicodedata.normalize("NFKD", strip_controls(str(text or "")))
    if not _COMBINING_SET.isdisjoint(text):
        text = _COMBINING_RE.sub("", text)
    text = text.lower()
    for final, regular in _FINALS:  # same as .translate(HEBREW_FINALS), faster on long texts
        if final in text:
            text = text.replace(final, regular)
    return " ".join(_TOKEN_RE.findall(text))


def _is_hebrew(token: str) -> bool:
    return any("֐" <= c <= "׿" for c in token)


def extract_terms(*texts: str) -> set[str]:
    """Meaningful search terms from free text (stopwords and bare numbers removed)."""
    terms: set[str] = set()
    for text in texts:
        for token in normalize_text(text).split():
            if token in STOPWORDS or token.isdigit():
                continue
            if _is_hebrew(token):
                if len(token) >= 2:
                    terms.add(token)
            elif len(token) >= 3:
                if len(token) > 4 and token.endswith("s"):
                    token = token[:-1]
                terms.add(token)
    return terms


def _hebrew_match(term: str, token: str) -> bool:
    """token is term with optional proclitic prefixes (ו/ה/ב/ל/מ/ש/כ) and a common inflection suffix."""
    stems = {term}
    if term.endswith("ה"):
        stems.add(term[:-1] + "ת")  # construct state: תקנה -> תקנת
        stems.add(term[:-1])  # before plural suffix: תקנה -> תקנות
    for stem in stems:
        idx = token.find(stem)
        if idx < 0:
            continue
        prefix, suffix = token[:idx], token[idx + len(stem):]
        if len(prefix) <= 3 and all(c in HEBREW_PREFIX_LETTERS for c in prefix) and suffix in HEBREW_SUFFIXES:
            return True
    return False


def _english_match(term: str, token: str) -> bool:
    return token.startswith(term) and len(token) - len(term) <= 3


def matched_terms(terms: set[str], text: str) -> set[str]:
    """Terms that occur in text, allowing Hebrew prefixes/suffixes and English inflections."""
    vocab = set(normalize_text(text).split())
    hits = set()
    for term in terms:
        if term in vocab:
            hits.add(term)
            continue
        hebrew = _is_hebrew(term)
        for tok in vocab:
            if (hebrew and _hebrew_match(term, tok)) or (not hebrew and _english_match(term, tok)):
                hits.add(term)
                break
    return hits


def excerpt_found(excerpt: str, content: str, min_chars: int = 15) -> bool:
    """True if a quoted excerpt appears in retrieved content (after normalization).

    Exact normalized substring first; otherwise at least 80% of 6-word windows must appear, which
    tolerates small PDF/table extraction differences without accepting paraphrases.
    """
    ex = normalize_text(excerpt)
    body = normalize_text(content)
    if len(ex) < min_chars or not body:
        return False
    if ex in body:
        return True
    words = ex.split()
    if len(words) < 6:
        return False
    windows = [" ".join(words[i : i + 6]) for i in range(0, len(words) - 5, 3)]
    found = sum(1 for w in windows if w in body)
    return found / len(windows) >= 0.8
