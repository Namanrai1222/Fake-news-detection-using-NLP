from sklearn.metrics import accuracy_score, classification_report

def evaluate(model, vectorizer, X_test, y_test):
    X_test_vec = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))