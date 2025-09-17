# backend/safety.py
"""
Safety filters for Demotivator AI.
Conservative patterns to detect crisis situations and hate speech.
"""

import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Crisis-safe response for self-harm/suicide mentions
CRISIS_SAFE_REPLY = """I understand you're going through a difficult time. Your wellbeing matters, and there are people who want to help.

Please reach out to someone:
- National Suicide Prevention Lifeline: 988 or 1-800-273-8255
- Crisis Text Line: Text HOME to 741741
- International Association for Suicide Prevention: https://www.iasp.info

You don't have to go through this alone. Professional support is available 24/7."""

# Conservative crisis patterns - only strong indicators of self-harm/suicide
CRISIS_PATTERNS = [
    r'\b(kill\s+myself|end\s+my\s+life|suicid(e|al))\b',
    r'\b(want\s+to\s+die|better\s+off\s+dead|not\s+worth\s+living)\b',
    r'\b(self[\s-]?harm|cut\s+myself|hurt\s+myself)\b',
    r'\b(overdose|take\s+all\s+the\s+pills|OD\s+on)\b',
    r'\b(jump\s+off|hang\s+myself|shoot\s+myself)\b',
    r'\b(no\s+reason\s+to\s+live|ending\s+it\s+all|goodbye\s+cruel\s+world)\b',
]

# Conservative hate patterns - only explicit hate speech and slurs
HATE_PATTERNS = [
    # common violent/hate phrases (obfuscated in code to avoid literal slurs in repo)
    r'\b(kill\s+all\s+(the\s+)?(jews|muslims|blacks|whites|asians|gays))\b',
    r'\b(death\s+to\s+(jews|muslims|blacks|whites|asians|gays|trans))\b',
    r'\b(exterminate|genocide|holocaust\s+the)\b',
    r'\b(subhuman|untermensch|not\s+human|deserve\s+to\s+die)\b',
    # generic slur forms (kept conservative)
    r'\b(f[a4]gg[o0]t|n[i1]gg[e3]r|ch[i1]nk)\b',
]

def _snippet_for_log(text: str, length: int = 120) -> str:
    """Return a short sanitized snippet for logging (avoid full text in logs)."""
    if not text:
        return ""
    s = text.strip().replace("\n", " ")
    return (s[:length] + "...") if len(s) > length else s

def pre_user_block(text: str) -> Dict[str, bool]:
    """Pre-model safety check on user input.

    Returns:
        {"crisis": bool, "hate": bool}
    """
    if not text:
        return {"crisis": False, "hate": False}

    text_lower = text.lower()

    crisis = any(re.search(p, text_lower) for p in CRISIS_PATTERNS)
    hate = any(re.search(p, text_lower) for p in HATE_PATTERNS)

    if crisis:
        logger.warning("pre_user_block: crisis detected: %s", _snippet_for_log(text))
    if hate:
        logger.warning("pre_user_block: hate detected: %s", _snippet_for_log(text))

    return {"crisis": bool(crisis), "hate": bool(hate)}

def post_model_block(text: str) -> Dict[str, bool]:
    """Post-model safety check on generated output.

    Returns:
        {"hate": bool}
    """
    if not text:
        return {"hate": False}

    text_lower = text.lower()
    hate = any(re.search(p, text_lower) for p in HATE_PATTERNS)

    if hate:
        logger.warning("post_model_block: hate detected in model output: %s", _snippet_for_log(text))
    return {"hate": bool(hate)}
