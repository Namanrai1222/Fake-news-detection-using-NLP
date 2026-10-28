import os
import sys
import threading
import traceback
import json
import logging
import re
import urllib.request
import urllib.error
import urllib.parse
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

app = Flask(__name__, static_folder=None)

FRONTEND_DIR = os.path.join(ROOT_DIR, 'frontend')

# Runtime logs work locally and on read-only serverless filesystems.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

# ── Global State ────────────────────────────────────────────────────────────
predict_fn = None
explain_fn = None
model_loading = True
model_error = None
_predictor_obj = None

OLLAMA_ENABLED = os.environ.get('OLLAMA_ENABLED', '0').lower() in ('1', 'true', 'yes', 'on')
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434/api/generate')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'tinyllama')
OLLAMA_TIMEOUT_SECONDS = float(os.environ.get('OLLAMA_TIMEOUT_SECONDS', '20'))
GOOGLE_FACTCHECK_API_KEY = os.environ.get('GOOGLE_FACTCHECK_API_KEY', '').strip()
APP_API_KEY = os.environ.get('APP_API_KEY', '').strip()
HEALTH_DETAILS = os.environ.get('HEALTH_DETAILS', '0').lower() in ('1', 'true', 'yes', 'on')
MAX_REQUEST_BYTES = int(os.environ.get('MAX_REQUEST_BYTES', str(512 * 1024)))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get('RATE_LIMIT_WINDOW_SECONDS', '60'))
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get('RATE_LIMIT_MAX_REQUESTS', '40'))
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        'ALLOWED_ORIGINS',
        'http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:5000,http://localhost:5000',
    ).split(',') if o.strip()
]

app.config['MAX_CONTENT_LENGTH'] = MAX_REQUEST_BYTES
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=False)

@app.after_request
def security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; connect-src 'self'; "
        "img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    )
    if request.is_secure:
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

_RATE_STATE = {}
_RATE_LOCK = threading.Lock()


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')

# Update to Version 2 from Config directly if available, else hardcode
# Since Predictor dynamically loads MODEL_PATH and VECTORIZER_PATH from config, 
# updating config.py earlier implicitly maps _v2 files for this load.

def _load_model():
    global predict_fn, explain_fn, model_loading, model_error, _predictor_obj
    try:
        from src.predict import Predictor
        
        print("[INFO] Initializing Predictor_v2.0...")
        predictor = Predictor()
        _predictor_obj = predictor
        
        if predictor.model is not None and predictor.vectorizer is not None:
            predict_fn = predictor.predict
            explain_fn = None
            model_error = None
            print(f"[OK] Core Engine Loaded. Vectorizer Features: {len(predictor.vectorizer.vocabulary_)}")
        else:
            model_error = predictor.load_error or "Model files not found. Run training script main.py first."
            print(f"[WARN] {model_error}")
    except Exception as e:
        model_error = f"Failed to load engine components: {e}"
        print(f"[ERROR] {model_error}")
        traceback.print_exc()
    finally:
        model_loading = False

_load_model()

# ── Helpers ─────────────────────────────────────────────────────────────────
BIAS_SOURCE_PATTERNS = {
    'reuters': [r'\breuters\b', r'\(reuters\)', r'\[reuters\]'],
    'associated press': [r'\bassociated\s+press\b'],
    'cnn': [r'\bcnn\b'],
    'fox news': [r'\bfox\s+news\b'],
    'bbc': [r'\bbbc\b'],
    # strict AP matching only as standalone source marker, not as substring.
    'ap': [r'\bap\b', r'\(ap\)', r'\[ap\]'],
}
POLITICAL_KEYWORDS = {
    'left', 'right', 'liberal', 'conservative', 'democrat', 'republican',
    'government', 'election', 'vote', 'propaganda', 'agenda',
}
EMOTION_WORDS = {
    'shocking', 'outrage', 'disaster', 'panic', 'fear', 'angry', 'furious',
    'massive', 'explosive', 'urgent', 'critical', 'alarming',
}
POSITIVE_TONE = {'improves', 'stabilized', 'growth', 'verified', 'official', 'confirmed'}
NEGATIVE_TONE = {'collapse', 'fraud', 'scam', 'fake', 'hoax', 'crisis', 'failure'}

def _check_bias(text: str):
    text_lower = text.lower()
    found = []
    for source, patterns in BIAS_SOURCE_PATTERNS.items():
        if any(re.search(pattern, text_lower) for pattern in patterns):
            found.append(source)

    found = sorted(set(found))
    tokens = re.findall(r"[a-z']+", text_lower)

    political_hits = sum(1 for t in tokens if t in POLITICAL_KEYWORDS)
    emotion_hits = sum(1 for t in tokens if t in EMOTION_WORDS)
    sentiment_score = sum(1 for t in tokens if t in POSITIVE_TONE) - sum(1 for t in tokens if t in NEGATIVE_TONE)

    risk_score = 0
    if len(found) > 1:
        risk_score += 2
    elif found:
        risk_score += 1
    if political_hits >= 3:
        risk_score += 1
    if emotion_hits >= 3:
        risk_score += 1

    if risk_score >= 3:
        risk_label = "High"
    elif risk_score >= 1:
        risk_label = "Medium"
    else:
        risk_label = "Low"

    return {
        "indicators": [f"Source reference detected: '{s.upper()}'" for s in found],
        "risk": risk_label,
        "political_density": political_hits,
        "emotion_density": emotion_hits,
        "sentiment_score": sentiment_score,
    }


def _error_response(message: str, code: int, extra: dict = None):
    payload = {
        "status": "error",
        "error": message,
    }
    if extra:
        payload.update(extra)
    return jsonify(payload), code


def _client_id():
    # Proxy headers are untrusted unless the deployment explicitly installs ProxyFix.
    return request.remote_addr or 'unknown'


def _is_rate_limited(client: str):
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    with _RATE_LOCK:
        timestamps = _RATE_STATE.get(client, [])
        timestamps = [t for t in timestamps if t >= cutoff]
        if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            _RATE_STATE[client] = timestamps
            return True
        timestamps.append(now)
        _RATE_STATE[client] = timestamps
    return False


def _require_api_access():
    client = _client_id()
    if _is_rate_limited(client):
        return _error_response("Too many requests. Please retry later.", 429)

    if APP_API_KEY:
        supplied = request.headers.get('X-API-Key', '').strip()
        if supplied != APP_API_KEY:
            return _error_response("Unauthorized API access.", 401)
    return None


def _extract_claim(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sentence in sentences:
        if len(sentence.split()) >= 8:
            return sentence[:300]
    return text[:300]


def _verify_claim(text: str):
    claim = _extract_claim(text)
    if not GOOGLE_FACTCHECK_API_KEY:
        return {
            "status": "not_configured",
            "provider": "google_fact_check",
            "claim": claim,
            "details": "Fact-check API key not configured.",
        }

    endpoint = (
        "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        f"?query={urllib.parse.quote(claim)}&key={GOOGLE_FACTCHECK_API_KEY}"
    )
    req = urllib.request.Request(endpoint, method='GET')
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected -- fixed Google HTTPS host
            body = resp.read().decode('utf-8')
            data = json.loads(body) if body else {}
            claims = data.get('claims', [])
            if not claims:
                return {
                    "status": "not_verified",
                    "provider": "google_fact_check",
                    "claim": claim,
                    "details": "No matching fact-check claim found.",
                }

            top = claims[0]
            review = (top.get('claimReview') or [{}])[0]
            return {
                "status": "verified_match_found",
                "provider": "google_fact_check",
                "claim": top.get('text', claim),
                "details": {
                    "publisher": (review.get('publisher') or {}).get('name'),
                    "url": review.get('url'),
                    "rating": review.get('textualRating'),
                },
            }
    except Exception as exc:
        logging.warning(f"Fact-check lookup unavailable; skipping. {exc}")
        return {
            "status": "unavailable",
            "provider": "google_fact_check",
            "claim": claim,
            "details": "Fact-check service unavailable at the moment.",
        }


def _build_reasoning(label: str, feature_signals, bias_info, verification):
    reasons = []
    if feature_signals:
        top = feature_signals[:3]
        terms = ", ".join([f"'{item['term']}'" for item in top])
        reasons.append(f"Top lexical evidence: {terms}.")

    if bias_info.get('emotion_density', 0) >= 3:
        reasons.append("Elevated emotional language detected, which can increase manipulation risk.")
    if bias_info.get('political_density', 0) >= 3:
        reasons.append("High political keyword density detected.")
    if verification.get('status') in {'not_verified', 'not_configured', 'unavailable'}:
        reasons.append("Claim could not be externally verified in this pass.")

    reasons.append(f"Final class chosen: {label}.")
    return reasons[:5]


def _latest_calibration_metrics():
    metrics_path = os.path.join(ROOT_DIR, "models", "metrics_v2.json")
    try:
        if not os.path.exists(metrics_path):
            return None, None
        with open(metrics_path, "r") as f:
            metrics = (json.load(f) or {}).get("model_metrics", {})
        ece = metrics.get("ece")
        brier = metrics.get("brier_score")
        return (float(ece) if ece is not None else None, float(brier) if brier is not None else None)
    except Exception:
        return None, None


def _build_ollama_prompt(text: str, label: str, confidence: float, bias_risk: str) -> str:
    preview = text[:1200]
    return (
        "You are assisting a fake-news classifier UI.\\n"
        f"Model prediction: {label}\\n"
        f"Confidence: {confidence:.4f}\\n"
        f"Bias risk: {bias_risk}\\n"
        "Task: Provide 2-3 short sentences explaining why this text might be labeled this way. "
        "Do not claim certainty. Mention linguistic cues and source framing patterns only.\\n"
        "Output plain text only.\\n\\n"
        f"Article excerpt:\\n{preview}"
    )


def _get_ollama_explanation(text: str, label: str, confidence: float, bias_risk: str, use_llm: bool, model_name: str = None):
    if not use_llm:
        return None

    parsed_url = urllib.parse.urlparse(OLLAMA_URL)
    if parsed_url.scheme != 'http' or parsed_url.hostname not in {'127.0.0.1', '::1'} or parsed_url.path != '/api/generate' or parsed_url.username or parsed_url.password:
        logging.warning('Ollama URL must point to the local generate endpoint.')
        return None

    prompt = _build_ollama_prompt(text, label, confidence, bias_risk)
    target_model = model_name or OLLAMA_MODEL
    payload = {
        "model": target_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 80,
        },
    }

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as resp:  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected -- loopback URL checked above
            body = resp.read().decode('utf-8')
            data = json.loads(body) if body else {}
            explanation = (data.get('response') or '').strip()
            return explanation or None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logging.warning(f"Ollama explanation unavailable; skipping. {exc}")
        return None
    except Exception as exc:
        logging.warning(f"Unexpected Ollama explanation failure; skipping. {exc}")
        return None

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    global explain_fn, _predictor_obj
    denied = _require_api_access()
    if denied:
        return denied

    if model_loading:
        return _error_response("System is currently initializing AI models. Please hold.", 503)
    # Treat model_error as fatal only when no usable predictor is available.
    # This avoids blocking requests on non-fatal warmup warnings.
    if model_error and predict_fn is None:
        return _error_response("Prediction engine unavailable.", 503)
    if not predict_fn:
        return _error_response("Predictor engine is unresponsive.", 503)

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or 'text' not in payload:
        return _error_response("Malformed request. Missing 'text' attribute.", 400)

    if not isinstance(payload['text'], str):
        return _error_response("'text' must be a string.", 400)
    input_text = payload['text'].strip()
    use_llm = _to_bool(payload.get('use_llm'), default=OLLAMA_ENABLED)
    llm_model = None
    
    # Advanced Input Validation
    if not input_text:
        return _error_response("The input text cannot be empty.", 400)
        
    word_count = len(input_text.split())
    if word_count < 20:
        return _error_response(
            f"Text too short ({word_count} words). Please provide at least 20 words for an accurate analysis.",
            400,
            {"code": "SHORT_INPUT"},
        )
        
    if word_count > 2500:
        return _error_response(
            f"Text too long ({word_count} words). Exceeds token limits.",
            400,
            {"code": "LONG_INPUT"},
        )

    if re.fullmatch(r"[^a-zA-Z0-9]+", input_text):
        return _error_response("Input appears non-linguistic or corrupted.", 400, {"code": "INVALID_INPUT"})

    try:
        if _predictor_obj is not None and hasattr(_predictor_obj, 'predict_with_details'):
            pred_details = _predictor_obj.predict_with_details(input_text)
            label = pred_details.get('label')
            confidence_raw = float(pred_details.get('confidence_raw', 0.0))
            confidence_calibrated = float(pred_details.get('confidence_calibrated', confidence_raw))
            feature_signals = pred_details.get('feature_signals', [])
        else:
            label, confidence_calibrated = predict_fn(input_text)
            confidence_raw = float(confidence_calibrated or 0.0)
            confidence_calibrated = float(confidence_calibrated or 0.0)
            feature_signals = []

        if not label:
            return _error_response("Prediction engine returned no class label.", 503)

        # Convert primary output to Real-class probability for blending.
        primary_real_prob = confidence_calibrated if label == 'Real' else (1.0 - confidence_calibrated)

        explanation = []
        if explain_fn:
            explanation = explain_fn(input_text)

        bias = _check_bias(input_text)
        verification = _verify_claim(input_text)

        # This is the calibrated model estimate. The old heuristic blend was not validated.
        final_label = label
        final_confidence = confidence_calibrated
        reliability_band = 'High' if final_confidence >= 0.8 else ('Medium' if final_confidence >= 0.6 else 'Low')
        reliability = {'band': reliability_band, 'score': round(final_confidence, 4)}

        reasoning = _build_reasoning(label, feature_signals, bias, verification)
        llm_explanation = _get_ollama_explanation(
            input_text,
            final_label,
            final_confidence,
            bias['risk'],
            use_llm=use_llm,
            model_name=llm_model,
        )
        
        # Log aggregate prediction metadata only.
        log_entry = (
            f"Len: {word_count} | Pred: {final_label} | "
            f"ConfRaw: {confidence_raw:.3f} | ConfCal: {confidence_calibrated:.3f} | "
            f"ConfFinal: {final_confidence:.3f} | Bias: {bias['risk']}"
        )
        logging.info(log_entry)

        return jsonify({
            "status": "success",
            "prediction":   final_label,
            "confidence":   round(float(final_confidence), 4),
            "confidence_raw": round(float(confidence_raw), 4),
            "confidence_calibrated": round(float(confidence_calibrated), 4),
            "explanation":  explanation,
            "feature_signals": feature_signals,
            "reasoning": reasoning,
            "reliability": reliability,
            "verification": verification,
            "llm_explanation": llm_explanation,
            "llm_provider": "ollama" if llm_explanation else None,
            "llm_requested": use_llm,
            "bias_analysis": bias,
            "error": None,
        })

    except Exception as exc:
        logging.error(f"Prediction crashed: {str(exc)}")
        traceback.print_exc()
        return _error_response("Analysis failed. Please try again.", 500)

@app.route('/stats', methods=['GET'])
def stats():
    denied = _require_api_access()
    if denied:
        return denied

    metrics_path = os.path.join(ROOT_DIR, "models", "metrics_v2.json")
    try:
        with open(metrics_path) as file:
            report = json.load(file)
        return jsonify({"status": "success", **report})
    except (OSError, ValueError):
        return _error_response("Model metrics unavailable.", 503)

@app.route('/health', methods=['GET'])
def health():
    health_status = "healthy"
    if model_loading:
        health_status = "initializing"
    elif model_error and predict_fn is None:
        health_status = "degraded"

    base = {
        "status": health_status,
        "model_loaded": predict_fn is not None,
    }

    if HEALTH_DETAILS:
        base.update({
            "model_loading": model_loading,
            "model_error": model_error,
            "ollama_default_enabled": OLLAMA_ENABLED,
            "ollama_default_model": OLLAMA_MODEL,
        })

    return jsonify(base)

@app.route('/')
def home():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(FRONTEND_DIR, path)

if __name__ == '__main__':
    print(f"[INFO] Backend API online. Target mapping UI: {FRONTEND_DIR}")
    run_host = os.environ.get('APP_HOST', '127.0.0.1')
    run_port = int(os.environ.get('APP_PORT', '5000'))
    run_debug = _to_bool(os.environ.get('APP_DEBUG', '0'), default=False)
    app.run(host=run_host, port=run_port, debug=run_debug, use_reloader=False)
