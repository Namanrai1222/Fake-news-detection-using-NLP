from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def train_model(X_train, y_train, model_path, vectorizer_path):
    vectorizer = TfidfVectorizer(max_features=5000)

    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    return model, vectorizer