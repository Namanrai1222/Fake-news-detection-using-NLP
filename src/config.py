import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAKE_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Fake.csv")
TRUE_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "True.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")