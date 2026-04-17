import os
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import clean_text
from src.config import BASE_DIR, FAKE_DATA_PATH, TRUE_DATA_PATH


def main():
    model_path = os.path.join(BASE_DIR, "models", "model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
    calibrator_path = os.path.join(BASE_DIR, "models", "calibrator.pkl")

    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
        raise FileNotFoundError("model.pkl/vectorizer.pkl not found in models/.")

    print("Loading existing artifacts...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    print("Loading data for calibration split...")
    data = load_data(FAKE_DATA_PATH, TRUE_DATA_PATH)
    data["cleaned"] = data["text"].apply(clean_text)
    data = data[data["cleaned"].str.len() > 0]

    # Use a held-out split only for calibration (no model refit)
    _, calib = train_test_split(
        data,
        test_size=0.15,
        random_state=42,
        stratify=data["label"],
    )

    X_calib = vectorizer.transform(calib["cleaned"])
    y_calib = calib["label"].to_numpy()

    print("Fitting isotonic calibrator on existing model outputs...")
    probs = model.predict_proba(X_calib)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probs, y_calib)

    raw_conf = np.maximum(probs, 1 - probs)
    cal_pos = calibrator.predict(probs)
    cal_conf = np.maximum(cal_pos, 1 - cal_pos)

    print(f"Mean raw confidence: {raw_conf.mean():.4f}")
    print(f"Mean calibrated confidence: {cal_conf.mean():.4f}")

    joblib.dump(calibrator, calibrator_path)
    print(f"Saved calibrator to {calibrator_path}")


if __name__ == "__main__":
    main()
