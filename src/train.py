from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib

def train_model(X_train, y_train, model_path, vectorizer_path):
    # Vectorizer is strictly fit on X_train to prevent data leakage
    # Reduced max_features slightly to curb overfitting shown by previous 99.2% stats
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    print("Fitting vectorizer on training data...")
    X_train_vec = vectorizer.fit_transform(X_train)

    # Added C=1.0 (default L2) and class_weight='balanced' to handle imbalance dynamically
    model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    
    # Run a quick 5-fold cross validation to objectively check training robustness
    print("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print("Fitting final model...")
    model.fit(X_train_vec, y_train)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"Model saved to {model_path}")

    return model, vectorizer