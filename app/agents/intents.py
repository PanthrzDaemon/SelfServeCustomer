"""Deterministic intent classification.

Rules are ordered by priority: safety-relevant intents (security incident,
human request, refund...) win over generic ones (troubleshooting, how-to). A
message that mentions nothing CloudFlow-related is out of scope.
"""

import re
from dataclasses import dataclass, field

INTENTS = (
    "prompt_injection", "unauthorized_request", "security_incident", "human_request", "data_privacy",
    "refund", "account_access", "password_reset", "api_rate_limit", "usage", "billing", "account_status",
    "platform_status", "troubleshooting", "how_to", "general_question", "out_of_scope",
)

# (intent, patterns). First intent with a matching pattern wins.
RULES: list[tuple[str, list[str]]] = [
    ("security_incident", [
        r"\bcompromi[sz]ed\b", r"\bhack(?:ed|er|ing)?\b", r"\bbreach(?:ed)?\b", r"\bleak(?:ed|ing)?\b",
        r"\bstolen\b", r"\bexposed\b", r"\bunauthori[sz]ed (?:access|login|logins|activity|user|charges?)\b",
        r"\bsuspicious (?:activity|login|logins|workflows?|members?)\b",
        r"\b(?:didn'?t|did not|never) (?:create|add|invite|authori[sz]e)\b", r"\bsomeone (?:else )?(?:is|has been) (?:using|logging|running)\b",
        r"\btaken over\b", r"\bphish",
    ]),
    ("human_request", [
        r"\b(?:talk|speak|chat) (?:to|with) (?:a |an |some )?(?:human|person|people|agent|someone|representative|rep)\b",
        r"\b(?:real|live) (?:person|human|agent)\b", r"\bhuman (?:agent|support|being)\b", r"\bescalate\b",
    ]),
    ("data_privacy", [
        r"\bgdpr\b", r"\bccpa\b", r"\bright to be forgotten\b", r"\bdata (?:deletion|erasure|removal)\b",
        r"\bdelete (?:all )?(?:my|our|their|his|her)?\s*(?:personal )?(?:data|information|details)\b",
        r"\bpersonal data\b",
    ]),
    ("refund", [r"\brefund", r"\bmoney back\b", r"\bchargeback\b", r"\breimburse"]),
    ("account_access", [
        r"\bowner (?:has )?left\b", r"\btransfer (?:the )?ownership\b", r"\bownership transfer\b", r"\blocked out\b",
        r"\blost (?:my |our )?(?:2fa|two[- ]factor|authenticator|phone|recovery codes?)\b",
        r"\b(?:can'?t|cannot|unable to) (?:access|get into) (?:my|our|the) (?:email|account|workspace)\b",
    ]),
    ("password_reset", [r"\bpassword\b", r"\breset link\b", r"\bforgot\b",
                        r"\b(?:can'?t|cannot|unable to)\s+(?:i\s+|we\s+)?(?:log|sign)\s*in\b",
                        r"\b(?:log|sign)[- ]?in (?:problem|issue|fail\w*)s?\b"]),
    ("api_rate_limit", [r"\b429s?\b", r"\brate[- ]?limit", r"\btoo many requests\b", r"\bthrottl", r"\bcf-429\b"]),
    ("usage", [r"\busage\b", r"\bquota\b", r"\bcf-402\b", r"\bhow many (?:workflow )?(?:runs|requests|api calls)\b",
               r"\b(?:over|under|near|hitting|reach(?:ed|ing)?|exceed(?:ed|ing)?) (?:my|our|the) (?:api )?limits?\b",
               r"\bapi limit\b", r"\brun limit\b"]),
    ("billing", [r"\bcharg(?:e|ed|es|ing)\b", r"\binvoice", r"\bpayment", r"\bbill(?:ed|ing)?\b", r"\bcredit card\b",
                 r"\bcard\b", r"\bpast[_ ]due\b", r"\bpric(?:e|ing)\b", r"\bsubscription\b", r"\bpaid\b"]),
    ("account_status", [r"\bsuspend", r"\bcf-403\b", r"\b403\b", r"\baccount status\b", r"\breactivat",
                        r"\bcancell?ed\b", r"\b(?:workflows?|nothing) (?:are |is )?(?:not|n't) running\b",
                        r"\bnone of (?:our|my) workflows\b"]),
    ("platform_status", [r"\boutage\b", r"\bstatus page\b", r"\bplatform status\b", r"\bsystem status\b",
                         r"\bis (?:the )?(?:cloudflow|api|dashboard|workflow engine|platform|service)\b.{0,20}\bdown\b",
                         r"\bdown (?:right now|today|currently)\b", r"\bincident\b"]),
    ("troubleshooting", [r"\berror", r"\bfail", r"\bcf-\d{3}\b", r"\btime[sd]? ?out", r"\bnot working\b", r"\bbroken\b",
                         r"\binvalid_grant\b", r"\bmissing\b", r"\bdoesn'?t work\b", r"\bstopped\b", r"\bissue\b",
                         r"\bproblem\b", r"\b401\b", r"\b50[0-9]\b", r"\bwhy (?:is|are|does|do|did|am)\b"]),
    ("how_to", [r"\bhow (?:do|can|to|should|would)\b", r"\bwhere (?:do|can|is)\b", r"\bcan (?:i|we)\b",
                r"\bset ?up\b", r"\bconfigur", r"\bcreate\b", r"\benable\b", r"\bconnect\b", r"\bexport\b",
                r"\bwhat (?:is|are|does)\b", r"\bdo you support\b", r"\bdoes cloudflow\b", r"\bis there\b"]),
]
_COMPILED = [(intent, [re.compile(p, re.IGNORECASE) for p in patterns]) for intent, patterns in RULES]

# Words that tie a message to CloudFlow support. Without any, the request is out of scope.
DOMAIN_TERMS = re.compile(
    r"\b(?:cloudflow|workflows?|runs?|steps?|triggers?|api|keys?|tokens?|rate|limits?|errors?|cf-\d+|integrat\w*|"
    r"connect\w*|salesforce|slack|github|google|sheets?|webhooks?|invoices?|bill\w*|charg\w*|payments?|cards?|"
    r"refunds?|plans?|upgrad\w*|downgrad\w*|subscription|pric\w*|accounts?|workspaces?|teams?|members?|seats?|"
    r"roles?|permissions?|password|log ?in|sign ?in|2fa|sso|saml|export\w*|history|usage|quota|timeouts?|retr\w*|"
    r"branch\w*|variables?|monitor\w*|alerts?|status|outage|down|support|human|agent|data|privacy|delete|security|"
    r"compromi\w+|hack\w*|cancel\w*|reactivat\w*|owner\w*|dashboard|version|4\.[23]|429|401|403|503|http|"
    r"automation|automate|schedul\w*|cron|zap\w*|sync\w*|apps?|tools?|features?|settings?)\b",
    re.IGNORECASE,
)


@dataclass
class IntentResult:
    intent: str
    matched: list[str] = field(default_factory=list)
    in_domain: bool = True


def classify_intent(message: str) -> IntentResult:
    for intent, patterns in _COMPILED:
        matched = [p.pattern for p in patterns if p.search(message)]
        if matched:
            return IntentResult(intent=intent, matched=matched)
    if DOMAIN_TERMS.search(message):
        return IntentResult(intent="general_question")
    return IntentResult(intent="out_of_scope", in_domain=False)


VERSION_MENTION = re.compile(r"\b(?:cloudflow\s*|version\s*|v)?(4\.[23])\b", re.IGNORECASE)


def mentioned_version(message: str) -> str | None:
    match = VERSION_MENTION.search(message)
    return match.group(1) if match else None


NEGATIVE = re.compile(r"\b(?:angry|furious|frustrat\w*|unacceptable|terrible|awful|ridiculous|worst|useless|"
                      r"disappointed|annoyed|upset|outrageous|scam)\b|!!", re.IGNORECASE)
POSITIVE = re.compile(r"\b(?:thanks|thank you|great|awesome|love|appreciate|perfect)\b", re.IGNORECASE)


def detect_sentiment(message: str) -> str:
    if NEGATIVE.search(message):
        return "negative"
    if POSITIVE.search(message):
        return "positive"
    return "neutral"


DISPUTE = re.compile(r"\b(?:shouldn'?t|should not|mistake|wrong(?:ly)?|unfair|dispute|incorrect|error on your|"
                     r"by mistake|not fair|why would you)\b", re.IGNORECASE)


def disputes(message: str) -> bool:
    return bool(DISPUTE.search(message))
