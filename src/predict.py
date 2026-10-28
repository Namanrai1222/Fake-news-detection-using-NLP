import os
import platform

# Set low thread counts before importing joblib/numpy to avoid OpenBLAS init failures.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import joblib
import numpy as np
from src.preprocessing import clean_text
from src.config import MODEL_PATH, VECTORIZER_PATH, BASE_DIR


def _apply_windows_platform_workaround():
    """Avoid rare WMI-related platform.machine() failures during scipy/sklearn import on Windows."""
    if os.name != 'nt':
        return
    fallback_arch = os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64")
    platform.machine = lambda: fallback_arch

class Predictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.calibrator = None
        self.load_error = None
        self.load_model()

    def load_model(self):
        _apply_windows_platform_workaround()

        calibrator_path = os.path.join(BASE_DIR, "models", "calibrator.pkl")
        if not all(os.path.exists(path) for path in (MODEL_PATH, VECTORIZER_PATH, calibrator_path)):
            self.load_error = "Trained model artifacts missing. Run python main.py."
            return
        try:
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            self.calibrator = joblib.load(calibrator_path)
        except Exception as exc:
            self.model = self.vectorizer = self.calibrator = None
            self.load_error = f"Model load failed: {exc}"

    def _calibrate_confidence(self, positive_prob: float, vec) -> float:
        calibrated = self.calibrator.predict_proba(vec)[0, 1]
        return float(np.clip(calibrated, 0.0, 1.0))

    def _feature_contributions(self, vec, top_k=8):
        """Return top weighted input features for linear models as local explanations."""
        if self.model is None or self.vectorizer is None:
            return []

        try:
            if not hasattr(self.model, "coef_"):
                return []

            coefs = self.model.coef_
            if coefs.ndim != 2 or coefs.shape[1] != vec.shape[1]:
                return []

            # Binary LR: coef_[0] corresponds to positive class direction.
            weights = coefs[0]
            values = vec.toarray()[0]
            contrib = values * weights

            feature_names = np.array(self.vectorizer.get_feature_names_out())
            non_zero_idx = np.where(values != 0)[0]
            if non_zero_idx.size == 0:
                return []

            sorted_idx = non_zero_idx[np.argsort(np.abs(contrib[non_zero_idx]))[::-1][:top_k]]
            items = []
            for idx in sorted_idx:
                impact = float(contrib[idx])
                items.append({
                    "term": str(feature_names[idx]),
                    "impact": impact,
                    "direction": "Real" if impact >= 0 else "Fake",
                })
            return items
        except Exception:
            return []

    def predict_with_details(self, text):
        if self.model is None or self.vectorizer is None:
            return {
                "label": None,
                "confidence_raw": 0.0,
                "confidence_calibrated": 0.0,
                "feature_signals": [],
            }

        cleaned = clean_text(text)
        vec = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vec)[0]
        probs = self.model.predict_proba(vec)[0]

        positive_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        pred_prob = float(max(probs))
        calibrated_positive_prob = self._calibrate_confidence(positive_prob, vec)

        label = "Real" if calibrated_positive_prob >= 0.5 else "Fake"
        if label == "Real":
            calibrated_conf = calibrated_positive_prob
        else:
            calibrated_conf = 1.0 - calibrated_positive_prob

        return {
            "label": label,
            "confidence_raw": pred_prob,
            "confidence_calibrated": float(np.clip(calibrated_conf, 0.0, 1.0)),
            "feature_signals": self._feature_contributions(vec, top_k=8),
        }

    def predict(self, text):
        details = self.predict_with_details(text)
        return details["label"], details["confidence_calibrated"]
