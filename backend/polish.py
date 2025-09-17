# backend/polish.py
"""
Text polishing utilities for Demotivator AI.
Removes advice, hedging, and ensures proper formatting.
"""

import re
from typing import Optional, Tuple, Dict

# Phrases that indicate advice or encouragement (to be removed)
ADVICE_PATTERNS = [
    r'\b(you\s+(should|could|might\s+want\s+to|need\s+to))\b',
    r'\b(try\s+(to|this|doing))\b',
    r'\b(here\'s\s+(how|what|why))\b',
    r'\b(I\s+(recommend|suggest|advise))\b',
    r'\b(consider\s+(doing|trying))\b',
    r'\b(it\'s\s+(important|crucial|essential)\s+to)\b',
    r'\b(make\s+sure\s+(to|you))\b',
    r'\b(don\'t\s+forget\s+to)\b',
    r'\b(remember\s+to)\b',
    r'\b(the\s+key\s+is)\b',
    r'\b(focus\s+on)\b',
    r'\b(start\s+(by|with))\b',
    r'\b(step\s+\d+:?)\b',
    r'\b(first,|second,|third,|finally,)\b',
]

# Hedging words and phrases (to be removed for more direct tone)
HEDGING_PATTERNS = [
    r'\b(maybe|perhaps|possibly|probably)\b',
    r'\b(could\s+be|might\s+be|may\s+be)\b',
    r'\b(somewhat|rather|quite|fairly)\b',
    r'\b(it\s+seems|appears\s+to\s+be)\b',
    r'\b(however|although|nevertheless)\b',
    r'\b(on\s+the\s+other\s+hand)\b',
    r'\b(that\s+being\s+said)\b',
]

# Simple word replacements for clarity
SIMPLE_MAP = {
    'utilize': 'use',
    'utilization': 'use',
    'implement': 'do',
    'implementation': 'doing',
    'facilitate': 'help',
    'endeavor': 'try',
    'commence': 'start',
    'terminate': 'end',
    'acquire': 'get',
    'demonstrate': 'show',
    'subsequently': 'then',
    'furthermore': 'also',
    'therefore': 'so',
    'notwithstanding': 'despite',
}

def word_count(text: str) -> int:
    """Count words in text.
    
    Args:
        text: Input text
        
    Returns:
        Number of words
    """
    if not text:
        return 0
    # Split on whitespace and filter empty strings
    words = [w for w in text.split() if w]
    return len(words)

def sanitize(text: str, target_words: Optional[int] = None) -> Tuple[str, Dict]:
    """Sanitize text to match Demotivator AI tone.
    
    Removes advice, hedging, and complex words.
    Ensures proper paragraph structure.
    
    Args:
        text: Input text to sanitize
        target_words: Target word count (not enforced, just for metrics)
        
    Returns:
        Tuple of (sanitized_text, metrics_dict)
    """
    if not text:
        return "", {"removed_advice": False, "removed_hedging": False, "words": 0}
    
    original_text = text
    metrics = {"removed_advice": False, "removed_hedging": False, "words": 0}
    
    # Remove advice patterns
    for pattern in ADVICE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            metrics["removed_advice"] = True
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove hedging patterns
    for pattern in HEDGING_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            metrics["removed_hedging"] = True
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Replace complex words with simple ones
    for complex_word, simple_word in SIMPLE_MAP.items():
        text = re.sub(r'\b' + complex_word + r'\b', simple_word, text, flags=re.IGNORECASE)
    
    # Clean up multiple spaces and empty lines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Split into sentences for paragraph restructuring
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Group sentences into 1-4 paragraphs
    if len(sentences) <= 3:
        # Single paragraph for very short responses
        paragraphs = [' '.join(sentences)]
    elif len(sentences) <= 8:
        # 2 paragraphs
        mid = len(sentences) // 2
        paragraphs = [
            ' '.join(sentences[:mid]),
            ' '.join(sentences[mid:])
        ]
    else:
        # 3-4 paragraphs for longer responses
        third = len(sentences) // 3
        paragraphs = [
            ' '.join(sentences[:third]),
            ' '.join(sentences[third:third*2]),
            ' '.join(sentences[third*2:])
        ]
    
    # Join paragraphs
    text = '\n\n'.join(p for p in paragraphs if p.strip())
    
    # Count final words
    metrics["words"] = word_count(text)
    
    return text.strip(), metrics

def ensure_verdict(text: str) -> str:
    """Ensure text ends with the required verdict line.
    
    Args:
        text: Input text
        
    Returns:
        Text with proper verdict line
    """
    if not text:
        text = "Your plan is doomed."
    
    # Remove any existing verdict-like lines
    verdict_patterns = [
        r'VERDICT:.*$',
        r'Success rate:.*$',
        r'Failure rate:.*$',
    ]
    
    for pattern in verdict_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Clean up trailing whitespace
    text = text.rstrip()
    
    # Add the standard verdict
    if not text.endswith('.') and not text.endswith('!') and not text.endswith('?'):
        text += '.'
    
    text += '\n\nVERDICT: Success rate: 0% | Failure rate: 100%'
    
    return text