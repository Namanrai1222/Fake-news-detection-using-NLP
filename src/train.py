"""Fit and save the article classifier."""

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_model(texts, labels, max_features=40000, c=1.0):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
        strip_accents='unicode',
        dtype=np.float32,
    )
    vectors = vectorizer.fit_transform(texts)
    model = LogisticRegression(C=c, max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(vectors, labels)
    return model, vectorizer


def save_model(model, vectorizer, model_path, vectorizer_path):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
