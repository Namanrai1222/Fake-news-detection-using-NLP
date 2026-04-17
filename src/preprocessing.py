"""
preprocessing.py – zero external dependencies at import time.

NLTK is loaded inside a background thread AFTER the module is first imported,
so the Flask server always starts in < 1 second.  Until NLTK is ready the
fallback (pure-Python) path is used transparently.
"""

import re
import string
import threading


# ── Minimal English stop-word list (no NLTK required) ──────────────────────
_STOP_WORDS_BASE = frozenset({
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than',
    'too','very','s','t','can','will','just','don','should','now','d','ll',
    'm','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn',
    'haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn',
    'weren','won','wouldn','said','also','would','could','may','might','shall',
    'get','got','go','goes','went','come','came','know','think','new','one',
    'two','three','make','made','time','year','years','people','way','day',
    'man','woman','government','state','country','president','us','u','like',
    'just','even','back','well','still','since',
})

# ── Mutable state ───────────────────────────────────────────────────────────
_stop_words  = set(_STOP_WORDS_BASE)
_lemmatize   = lambda w: w   # identity until NLTK is ready
_tokenize    = lambda s: s.split()
_nltk_ready  = threading.Event()


def _load_nltk_async():
    """Background thread: loads NLTK resources and upgrades the globals."""
    global _stop_words, _lemmatize, _tokenize
    try:
        import nltk  # may take a few seconds the first time

        # ---- stop words ----
        for _pkg, _path in [('stopwords', 'corpora/stopwords')]:
            try:
                nltk.data.find(_path)
            except LookupError:
                nltk.download(_pkg, quiet=True, raise_on_error=False)
        try:
            from nltk.corpus import stopwords
            _stop_words = set(stopwords.words('english')) | _STOP_WORDS_BASE
        except Exception:
            pass

        # ---- tokeniser ----
        for _pkg, _path in [('punkt_tab', 'tokenizers/punkt_tab'),
                             ('punkt',     'tokenizers/punkt')]:
            try:
                nltk.data.find(_path)
                break
            except LookupError:
                nltk.download(_pkg, quiet=True, raise_on_error=False)
        try:
            from nltk.tokenize import word_tokenize
            word_tokenize('test')   # smoke-test
            _tokenize = word_tokenize
        except Exception:
            pass

        # ---- lemmatiser ----
        for _pkg, _path in [('wordnet', 'corpora/wordnet')]:
            try:
                nltk.data.find(_path)
            except LookupError:
                nltk.download(_pkg, quiet=True, raise_on_error=False)
        try:
            from nltk.stem import WordNetLemmatizer
            _lem = WordNetLemmatizer()
            _lemmatize = _lem.lemmatize
        except Exception:
            pass

        print("[NLP] NLTK resources loaded successfully.")
    except Exception as exc:
        print(f"[NLP] NLTK not available – using pure-Python fallback. ({exc})")
    finally:
        _nltk_ready.set()


# Kick off in background immediately – server is not blocked
threading.Thread(target=_load_nltk_async, daemon=True, name="nltk-loader").start()


# ── Public API ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Forensic text cleaner.
    Always returns quickly regardless of NLTK status.
    Once NLTK loads in the background, subsequent calls use the full pipeline.
    """
    if not isinstance(text, str):
        return ""

    # Remove Reuters / AP attribution headers (dataset-specific bias)
    text = re.sub(r'^.*?\(Reuters\)\s*[-–]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^.*?\(AP\)\s*[-–]',      '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?\]', '', text)    # bracketed source tags

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = _tokenize(text)
    words = [
        _lemmatize(w)
        for w in tokens
        if isinstance(w, str) and w.isalpha() and len(w) > 1 and w not in _stop_words
    ]
    return ' '.join(words)