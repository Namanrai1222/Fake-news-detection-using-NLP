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

# Setup basic file logging for predictions
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'predictions.log'),
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

# ── Global State ────────────────────────────────────────────────────────────
predict_fn = None
explain_fn = None
model_loading = True
model_error = None
_predictor_obj = None
_model_ready_event = threading.Event()

OLLAMA_ENABLED = os.environ.get('OLLAMA_ENABLED', '0').lower() in ('1', 'true', 'yes', 'on')
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434/api/generate')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'tinyllama')
OLLAMA_TIMEOUT_SECONDS = float(os.environ.get('OLLAMA_TIMEOUT_SECONDS', '20'))
GOOGLE_FACTCHECK_API_KEY = os.environ.get('GOOGLE_FACTCHECK_API_KEY', '').strip()
ENSEMBLE_ENABLED = os.environ.get('ENSEMBLE_ENABLED', '1').lower() in ('1', 'true', 'yes', 'on')
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

def _load_model_async():
    global predict_fn, explain_fn, model_loading, model_error, _predictor_obj
    try:
        from src.predict import Predictor
        
        print("[INFO] Initializing Predictor_v2.0...")
        predictor = Predictor()
        _predictor_obj = predictor
        
        if predictor.model is not None and predictor.vectorizer is not None:
            predict_fn = predictor.predict
            explain_fn = None
            print(f"[OK] Core Engine Loaded. Vectorizer Features: {len(predictor.vectorizer.vocabulary_)}")
        else:
            model_error = predictor.load_error or "Model files not found. Run training script main.py first."
            print(f"[WARN] {model_error}")
    except Exception as e:
        model_error = f"Failed to load engine components: {e}"
        print(f"[ERROR] {model_error}")
        traceback.print_exc()
    finally:
        _model_ready_event.set()
        model_loading = False

threading.Thread(target=_load_model_async, daemon=True, name="model-loader").start()


def _model_watchdog(timeout_seconds: int = 20):
    """Fail fast if model loading blocks unexpectedly on dependency import/load."""
    global model_loading, model_error
    if _model_ready_event.wait(timeout_seconds):
        return
    model_error = "Model initialization timed out. Check Python/scipy/sklearn compatibility and model files."
    model_loading = False
    logging.error(model_error)


threading.Thread(target=_model_watchdog, daemon=True, name="model-watchdog").start()

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
    # Respect proxy-forwarded IP when available, else fallback to remote_addr.
    forwarded = request.headers.get('X-Forwarded-For', '').split(',')[0].strip()
    return forwarded or (request.remote_addr or 'unknown')


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
        with urllib.request.urlopen(req, timeout=8) as resp:
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


def _secondary_model_score(text: str, bias_info, feature_signals):
    """Lightweight secondary scorer used for ensemble blending.

    Returns a probability for the Real class in [0, 1].
    """
    text_lower = text.lower()

    # Start from neutral.
    real_prob = 0.5
    reasons = []

    # Emotional spikes often correlate with misinformation style.
    emotion_density = int(bias_info.get('emotion_density', 0))
    if emotion_density >= 3:
        real_prob -= 0.12
        reasons.append("High emotional language reduced trust score.")

    # Presence of source markers can slightly increase traceability.
    source_hits = len(bias_info.get('indicators', []) or [])
    if source_hits >= 1:
        real_prob += 0.06
        reasons.append("Detected source references increased traceability score.")

    # Very high political density increases manipulation risk.
    political_density = int(bias_info.get('political_density', 0))
    if political_density >= 4:
        real_prob -= 0.08
        reasons.append("Dense political framing reduced trust score.")

    # If top features are dominated by domain terms, slightly increase confidence.
    domain_terms = {
        'inflation', 'policy', 'central bank', 'rate', 'economy', 'official',
        'statement', 'report', 'regulator', 'ministry'
    }
    top_terms = [str(x.get('term', '')).lower() for x in (feature_signals or [])[:6]]
    if any(any(dt in term for dt in domain_terms) for term in top_terms):
        real_prob += 0.06
        reasons.append("Domain-specific terminology increased trust score.")

    # Clickbait patterns reduce trust.
    clickbait_patterns = ["you won't believe", "shocking", "breaking", "must see", "viral"]
    if any(p in text_lower for p in clickbait_patterns):
        real_prob -= 0.12
        reasons.append("Clickbait phrase pattern reduced trust score.")

    real_prob = float(max(0.02, min(0.98, real_prob)))
    label = 'Real' if real_prob >= 0.5 else 'Fake'
    confidence = real_prob if label == 'Real' else (1.0 - real_prob)
    return {
        'real_probability': real_prob,
        'label': label,
        'confidence': round(float(confidence), 4),
        'reasons': reasons[:3],
    }


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
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as resp:
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
    if model_error:
        return _error_response(f"Engine critical failure: {model_error}", 500)
    if not predict_fn:
        return _error_response("Predictor engine is unresponsive.", 503)

    payload = request.get_json(silent=True)
    if not payload or 'text' not in payload:
        return _error_response("Malformed request. Missing 'text' attribute.", 400)

    input_text = payload['text'].strip()
    use_llm = _to_bool(payload.get('use_llm'), default=OLLAMA_ENABLED)
    llm_model = str(payload.get('llm_model', '')).strip() or None
    use_ensemble = _to_bool(payload.get('use_ensemble'), default=ENSEMBLE_ENABLED)
    
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

        # Lazy-load optional LIME explainer only after model is already ready.
        # This prevents startup from getting stuck when optional explainability deps are slow.
        if explain_fn is None and _predictor_obj is not None:
            try:
                from src.explain.lime_explainer import NewsExplainer
                explainer = NewsExplainer(_predictor_obj.model, _predictor_obj.vectorizer)
                explain_fn = explainer.explain
                explanation = explain_fn(input_text)
            except Exception as lime_exc:
                logging.warning(f"LIME unavailable; returning prediction without explanation. {lime_exc}")

        bias = _check_bias(input_text)
        verification = _verify_claim(input_text)

        secondary = _secondary_model_score(input_text, bias, feature_signals)
        final_label = label
        final_confidence = confidence_calibrated
        ensemble_info = {
            'enabled': bool(use_ensemble),
            'weights': {'primary': 0.6, 'secondary': 0.4},
            'primary_real_probability': round(float(primary_real_prob), 4),
            'secondary_real_probability': round(float(secondary['real_probability']), 4),
            'secondary_label': secondary['label'],
            'secondary_confidence': secondary['confidence'],
            'secondary_reasons': secondary['reasons'],
        }

        if use_ensemble:
            blended_real = (0.6 * primary_real_prob) + (0.4 * secondary['real_probability'])
            final_label = 'Real' if blended_real >= 0.5 else 'Fake'
            final_confidence = blended_real if final_label == 'Real' else (1.0 - blended_real)
            ensemble_info['blended_real_probability'] = round(float(blended_real), 4)
        else:
            ensemble_info['blended_real_probability'] = round(float(primary_real_prob), 4)

        ece_metric, _ = _latest_calibration_metrics()
        reliability_score = float(final_confidence)
        if ece_metric is not None:
            reliability_score = max(0.0, min(1.0, reliability_score - (0.5 * ece_metric)))

        if reliability_score >= 0.8:
            reliability_band = "High"
        elif reliability_score >= 0.6:
            reliability_band = "Medium"
        else:
            reliability_band = "Low"

        reliability = {
            "band": reliability_band,
            "score": round(float(reliability_score), 4),
            "ece_used": round(float(ece_metric), 4) if ece_metric is not None else None,
        }

        reasoning = _build_reasoning(label, feature_signals, bias, verification)
        if use_ensemble and secondary['reasons']:
            reasoning.extend(secondary['reasons'])
            reasoning = reasoning[:6]
        llm_explanation = _get_ollama_explanation(
            input_text,
            final_label,
            final_confidence,
            bias['risk'],
            use_llm=use_llm,
            model_name=llm_model,
        )
        
        # Log to file
        log_entry = (
            f"Len: {word_count} | Pred: {final_label} | "
            f"ConfRaw: {confidence_raw:.3f} | ConfCal: {confidence_calibrated:.3f} | "
            f"ConfFinal: {final_confidence:.3f} | Bias: {bias['risk']} | Ensemble: {use_ensemble}"
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
            "ensemble": ensemble_info,
            "reliability": reliability,
            "verification": verification,
            "llm_explanation": llm_explanation,
            "llm_provider": "ollama" if llm_explanation else None,
            "llm_requested": use_llm,
            "ensemble_requested": use_ensemble,
            "bias_analysis": bias,
            "error": None,
        })

    except Exception as exc:
        logging.error(f"Prediction crashed: {str(exc)}")
        traceback.print_exc()
        return _error_response(f"Analysis failed: {str(exc)}", 500)

@app.route('/stats', methods=['GET'])
def stats():
    denied = _require_api_access()
    if denied:
        return denied

    # Attempt to load the dynamically created stats from Model Training sequence
    metrics_path = os.path.join(ROOT_DIR, "models", "metrics_v2.json")
    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
                return jsonify({
                    "status": "success",
                    "total_articles": 44898,
                    "model_metrics": metrics_data.get("model_metrics", {}),
                    "training_date": datetime.now().strftime("%Y-%m-%d"),
                    "error": None,
                })
    except Exception as e:
        print("Failed reading true metrics:", e)
        
    # Fallback structure if models haven't been trained yet
    return jsonify({
        "status": "success",
        "total_articles": 44898,
        "model_metrics": {
            "accuracy":      0.0,
            "precision":     0.0,
            "recall":        0.0,
            "f1_score":      0.0,
            "ece":           0.0,
            "brier_score":   0.0,
            "confusion_matrix": [[0,0],[0,0]]
        },
        "training_date": "N/A",
        "error": None,
    })

@app.route('/health', methods=['GET'])
def health():
    base = {
        "status": "healthy",
        "model_loaded": predict_fn is not None,
    }

    if HEALTH_DETAILS:
        base.update({
            "model_loading": model_loading,
            "model_error": model_error,
            "ollama_default_enabled": OLLAMA_ENABLED,
            "ollama_default_model": OLLAMA_MODEL,
            "ensemble_default_enabled": ENSEMBLE_ENABLED,
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
