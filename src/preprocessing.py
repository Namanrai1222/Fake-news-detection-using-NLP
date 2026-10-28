"""Deterministic text normalization shared by training and inference."""

import re

# Keep negation: removing it changes the meaning of factual claims.
_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'as', 'at', 'by', 'for',
    'from', 'in', 'into', 'of', 'on', 'to', 'with', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'it', 'its', 'this', 'that', 'these',
    'those', 'there', 'their', 'they', 'them', 'he', 'she', 'his', 'her',
    'we', 'our', 'you', 'your', 'i', 'my', 'me', 'who', 'which', 'what',
    'when', 'where', 'how', 'said', 'also', 'would', 'could', 'will',
})
_ATTRIBUTION = re.compile(r'^.{0,100}?\((?:reuters|ap)\)\s*[-–]', re.IGNORECASE)
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = _ATTRIBUTION.sub('', text.strip())
    text = re.sub(r'\[[^]]{1,80}\]', '', text).lower()
    return ' '.join(word for word in _TOKEN.findall(text) if word not in _STOP_WORDS)
