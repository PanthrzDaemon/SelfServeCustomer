"""Detect and redact secrets and personal data.

Findings record only the *type* of what was found, never the value, so they are
safe to put in audit logs and handoffs.
"""

import re
from dataclasses import dataclass, field

SECRET = "secret"
PII = "pii"


@dataclass(frozen=True)
class Pattern:
    kind: str
    category: str
    regex: re.Pattern
    # Replace only this capture group (keeps e.g. "password:" visible).
    group: int = 0


def _p(kind: str, category: str, pattern: str, group: int = 0, flags: int = 0) -> Pattern:
    return Pattern(kind, category, re.compile(pattern, flags), group)


# Order matters: secrets first, so e.g. a token isn't half-matched as a phone number.
PATTERNS: list[Pattern] = [
    _p("private_key", SECRET, r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", flags=re.S),
    _p("api_key", SECRET, r"\bcf_(?:live|test)_[A-Za-z0-9]{8,}\b"),
    _p("api_key", SECRET, r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{8,}\b"),
    _p("api_key", SECRET, r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    _p("api_key", SECRET, r"\bAKIA[0-9A-Z]{16}\b"),
    _p("token", SECRET, r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    _p("token", SECRET, r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    _p("token", SECRET, r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"),
    _p("token", SECRET, r"(?i)\bbearer\s+([A-Za-z0-9._~+/-]{12,}=*)", group=1),
    # "password: X" / "password = X" always; "password is X" only when X looks like a secret
    # (has a digit or symbol), so "my password is expired" is left alone.
    _p("password", SECRET, r"(?i)\b(?:password|passwd|pwd|passcode|passphrase)\b\s*[:=]\s*[\"']?([^\s\"',;]{3,})", group=1),
    _p("password", SECRET,
       r"(?i)\b(?:password|passwd|pwd|passcode|passphrase)\b\s+is\s+[\"']?((?=[^\s\"',;]*[\d!@#$%^&*])[^\s\"',;]{4,})",
       group=1),
    _p("credential", SECRET,
       r"(?i)\b(?:api[ _-]?key|secret|access[ _-]?token|auth[ _-]?token|token|client[ _-]?secret|security[ _-]?token)"
       r"\b\s*(?:[:=]|\s+is)\s*[\"']?((?=[A-Za-z0-9._~+/=-]*\d)[A-Za-z0-9._~+/=-]{8,})", group=1),
    _p("card_security_code", PII, r"(?i)\b(?:cvv|cvc|cvv2|security code)\b\s*:?\s*(\d{3,4})\b", group=1),
    _p("payment_card", PII, r"\b(?:\d[ -]?){12,18}\d\b"),
    _p("ssn", PII, r"\b\d{3}-\d{2}-\d{4}\b"),
    _p("email", PII, r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    _p("phone", PII, r"(?<![\w-])(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b"),
]

# Company addresses that appear in public documentation are not personal data.
PUBLIC_EMAIL_DOMAINS = ("cloudflow.example",)


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass
class RedactionResult:
    text: str
    findings: list[dict] = field(default_factory=list)

    @property
    def has_secrets(self) -> bool:
        return any(f["category"] == SECRET for f in self.findings)

    @property
    def has_pii(self) -> bool:
        return any(f["category"] == PII for f in self.findings)

    @property
    def kinds(self) -> list[str]:
        return sorted({f["kind"] for f in self.findings})


def _skip(pattern: Pattern, value: str) -> bool:
    if pattern.kind == "payment_card":
        return not _luhn_ok(value)
    if pattern.kind == "email":
        return value.lower().endswith(PUBLIC_EMAIL_DOMAINS)
    if "[REDACTED" in value:
        return True
    return False


def redact(text: str) -> RedactionResult:
    findings: list[dict] = []
    for pattern in PATTERNS:
        def replace(match: re.Match, pattern: Pattern = pattern) -> str:
            value = match.group(pattern.group)
            if value is None or _skip(pattern, value):
                return match.group(0)
            findings.append({"kind": pattern.kind, "category": pattern.category})
            token = f"[REDACTED_{pattern.kind.upper()}]"
            if pattern.group == 0:
                return token
            start, end = match.span(pattern.group)
            offset = match.start(0)
            whole = match.group(0)
            return whole[: start - offset] + token + whole[end - offset:]

        text = pattern.regex.sub(replace, text)
    return RedactionResult(text=text, findings=findings)


def contains_sensitive(text: str) -> list[str]:
    """Kinds of secrets/PII present in text (used by the critic on answers)."""
    return redact(text).kinds
