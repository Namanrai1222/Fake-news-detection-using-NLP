import joblib
import numpy as np
import os
from src.explain.lime_explainer import NewsExplainer
from src.config import MODEL_PATH, VECTORIZER_PATH

def debug():
    if not os.path.exists(MODEL_PATH):
        print("Model not found")
        return
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    print("Model classes:", model.classes_)
    
    explainer = NewsExplainer(model, vectorizer)
    
    test_text = "This is a fake news about aliens and portals."
    probs = explainer.predict_probs([test_text])
    print("Probabilities:", probs)
    
    pred = np.argmax(probs[0])
    print("Prediction index:", pred)
    
    try:
        explanation = explainer.explain(test_text)
        print("Explanation success:", explanation)
    except Exception as e:
        import traceback
        print("Explanation failed!")
        traceback.print_exc()

if __name__ == "__main__":
    debug()
