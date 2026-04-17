from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)
import json
import os
import numpy as np
from src.config import BASE_DIR


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error for binary probabilities."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        acc_bin = np.mean(y_true[mask])
        conf_bin = np.mean(y_prob[mask])
        ece += np.abs(acc_bin - conf_bin) * (np.sum(mask) / len(y_prob))

    return float(ece)

def evaluate(model, vectorizer, X_test, y_test):
    # Vectorizer transform strictly on X_test (No fitting!)
    X_test_vec = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vec)
    
    # Calculate comprehensive metrics
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    y_prob_real = model.predict_proba(X_test_vec)[:, 1]
    brier = brier_score_loss(y_test, y_prob_real)
    ece = expected_calibration_error(y_test, y_prob_real, n_bins=10)

    print("\n--- FINAL MODEL EVALUATION ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Brier:     {brier:.4f}")
    print(f"ECE(10):   {ece:.4f}")
    print("\nConfusion Matrix:")
    print(f"[{cm[0][0]}  {cm[0][1]}]")
    print(f"[{cm[1][0]}  {cm[1][1]}]")
    print("\nDetailed Report:\n", classification_report(y_test, predictions, target_names=["Fake", "Real"]))
    
    # Save the realistic stats to a JSON file so the backend /stats route can load it dynamically
    stats_path = os.path.join(BASE_DIR, "models", "metrics_v2.json")
    metrics_data = {
        "model_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "brier_score": round(brier, 4),
            "ece": round(ece, 4),
            "confusion_matrix": cm.tolist()
        }
    }
    with open(stats_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Metrics saved to {stats_path}")