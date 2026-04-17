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

        candidate_pairs = [
            (MODEL_PATH, VECTORIZER_PATH),
            (os.path.join(BASE_DIR, "models", "model.pkl"), os.path.join(BASE_DIR, "models", "vectorizer.pkl")),
            (os.path.join(BASE_DIR, "models", "model_v2.pkl"), os.path.join(BASE_DIR, "models", "vectorizer_v2.pkl")),
        ]

        selected_pair = next(
            ((m, v) for (m, v) in candidate_pairs if os.path.exists(m) and os.path.exists(v)),
            None,
        )

        if not selected_pair:
            self.load_error = (
                "Model files not found. Expected one of these pairs: "
                + "; ".join([f"({m}, {v})" for (m, v) in candidate_pairs])
            )
            print(self.load_error)
            return

        model_path, vectorizer_path = selected_pair
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print(f"Loaded model from: {model_path}")
            print(f"Loaded vectorizer from: {vectorizer_path}")

            calibrator_path = os.path.join(BASE_DIR, "models", "calibrator.pkl")
            if os.path.exists(calibrator_path):
                self.calibrator = joblib.load(calibrator_path)
                print(f"Loaded calibrator from: {calibrator_path}")
        except Exception as exc:
            self.load_error = f"Model load failed for pair ({model_path}, {vectorizer_path}): {exc}"
            print(self.load_error)

    def _calibrate_confidence(self, positive_prob: float) -> float:
        if self.calibrator is None:
            return float(positive_prob)
        try:
            calibrated = self.calibrator.predict(np.array([positive_prob]))[0]
            return float(np.clip(calibrated, 0.0, 1.0))
        except Exception:
            return float(positive_prob)

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
        calibrated_positive_prob = self._calibrate_confidence(positive_prob)

        label = "Real" if prediction == 1 else "Fake"
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
