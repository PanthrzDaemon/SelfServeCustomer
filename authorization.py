

import re

ACCOUNT_ID = re.compile(r"\bACC-\d{4}\b", re.IGNORECASE)

CROSS_ACCOUNT_PHRASES = [
    r"\b(?:another|other|a different) (?:customer|company|client|organi[sz]ation)s?(?:'s|')?\b",
    r"\b(?:another|other|a different|someone else'?s?) (?:user|account|workspace|team)'?s? "
    r"(?:invoices?|usage|billing|bills?|data|details|information|info|payments?|cards?|plan|subscription|history)\b",
    r"\bsomeone else'?s\b",
    r"\b(?:all|every|list (?:of )?(?:all )?)\s*(?:customers|accounts|workspaces|companies)\b",
    r"\b(?:competitor|their) (?:account|invoices?|usage|billing|data)\b",
]
_PHRASES = [re.compile(p, re.IGNORECASE) for p in CROSS_ACCOUNT_PHRASES]


def other_account_ids(message: str, own_account_id: str) -> list[str]:
    return sorted({m.upper() for m in ACCOUNT_ID.findall(message)} - {own_account_id.upper()})


def requests_other_customer_data(message: str, own_account_id: str) -> str | None:
    """Return a reason string if the message asks for another customer's data."""
    if other_account_ids(message, own_account_id):
        return "references another account ID"
    for phrase in _PHRASES:
        if phrase.search(message):
            return "asks for another customer's data"
    return None
