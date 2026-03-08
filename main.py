from src.data_loader import load_data
from src.preprocessing import clean_text
from src.train import train_model
from src.evaluate import evaluate
from src.config import FAKE_DATA_PATH, TRUE_DATA_PATH, MODEL_PATH, VECTORIZER_PATH
from sklearn.model_selection import train_test_split


def main():
    print("Loading data...")
    data = load_data(FAKE_DATA_PATH, TRUE_DATA_PATH)

    print("Cleaning text...")
    data["cleaned"] = data["text"].apply(clean_text)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        data["cleaned"],
        data["label"],
        test_size=0.2,
        random_state=42
    )

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