import re

INJECTION_PATTERNS = [
    r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,30}\b(?:instructions?|prompts?|guidelines|"
    r"(?:your|system|previous|prior|above|earlier) (?:rules|polic(?:y|ies)))\b",
    r"\b(?:reveal|show|print|repeat|output|display|leak)\b[^.\n]{0,30}\b(?:system|hidden|initial|developer|internal)\s+"
    r"(?:prompt|instructions?|message|rules)\b",
    r"\byou are now\b",
    r"\b(?:developer|god|admin|jailbreak|dan) mode\b",
    r"\bact as\b[^.\n]{0,40}\b(?:without|no)\b[^.\n]{0,20}\b(?:restrictions|rules|limits|filters)\b",
    r"\bnew (?:system )?instructions?\s*:",
    r"^\s*(?:system|assistant)\s*:",
    r"<\s*/?\s*(?:system|assistant|instructions?)\s*>",
    r"###\s*(?:system|instruction)",
    r"\bpretend (?:that )?you (?:are|have)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in INJECTION_PATTERNS]


def detect_injection(text: str) -> list[str]:
    """Return the patterns matched (empty list = no injection detected)."""
    return [p.pattern[:40] for p in _COMPILED if p.search(text)]


def is_suspicious(text: str) -> bool:
    return bool(detect_injection(text))


def escape_untrusted(text: str) -> str:
    """Neutralize anything that could close or forge our <source> delimiters.

    Escaping "<" is enough to stop tags; ">" stays readable because the docs use
    it for navigation paths such as "Settings > Billing".
    """
    return text.replace("<", "&lt;")
