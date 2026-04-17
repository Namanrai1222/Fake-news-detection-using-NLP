from src.data_loader import load_data
from src.preprocessing import clean_text
from src.train import train_model
from src.evaluate import evaluate
from src.config import FAKE_DATA_PATH, TRUE_DATA_PATH, MODEL_PATH, VECTORIZER_PATH
from sklearn.model_selection import train_test_split


def main():
    print("Loading and cleaning data...")
    data = load_data(FAKE_DATA_PATH, TRUE_DATA_PATH)

    print("Cleaning text (this may take a few minutes)...")
    data["cleaned"] = data["text"].apply(clean_text)

    # After cleaning, some texts might become empty or very short. Filter again just in case
    initial_len = len(data)
    data = data[data["cleaned"].str.len() > 0]
    print(f"Dropped {initial_len - len(data)} records that became empty after strict cleaning.")

    print("Splitting dataset (Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        data["cleaned"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"]  # MANDATORY FIX: Stratification prevents class imbalance leakage
    )

    print(f"Training split: {len(X_train)} samples")
    print(f"Testing split: {len(X_test)} samples")

    print("Training TF-IDF model...")
    model, vectorizer = train_model(
        X_train,
        y_train,
        MODEL_PATH,
        VECTORIZER_PATH
    )

    print("Evaluating model...")
    evaluate(model, vectorizer, X_test, y_test)


if __name__ == "__main__":
    main()