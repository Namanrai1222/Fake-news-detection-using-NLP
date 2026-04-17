import numpy as np
from lime.lime_text import LimeTextExplainer
from src.preprocessing import clean_text   # ← FIX: clean before vectorizing


class NewsExplainer:
    def __init__(self, model, vectorizer, class_names=None):
        # class_names order MUST match model.classes_ order
        # Logistic Regression trained with label 0=Fake, 1=Real
        self.class_names = class_names or ['Fake', 'Real']
        self.model = model
        self.vectorizer = vectorizer
        self.explainer = LimeTextExplainer(class_names=self.class_names)

    def predict_probs(self, texts):
        """
        LIME requires a function: list[str] → ndarray(n, n_classes).
        We MUST clean text here the same way training did, otherwise
        the vectorizer maps entirely wrong features.
        """
        cleaned = [clean_text(t) for t in texts]   # ← KEY FIX
        vec = self.vectorizer.transform(cleaned)
        return self.model.predict_proba(vec)

    def explain(self, text: str, num_features: int = 8):
        """
        Returns a list of dicts: [{word, impact}, …]
        Positive impact → pushes towards Real; negative → pushes towards Fake.
        """
        # Determine predicted label index BEFORE calling LIME
        predicted_probs = self.predict_probs([text])[0]
        predicted_label = int(np.argmax(predicted_probs))

        try:
            exp = self.explainer.explain_instance(
                text,
                self.predict_probs,
                num_features=num_features,
                labels=(predicted_label,),   # ← Explicit labels avoids KeyError
            )
            explanation_list = exp.as_list(label=predicted_label)
        except Exception as e:
            print(f"[LIME] Explanation failed: {e}")
            return []

        return [
            {"word": str(word), "impact": float(weight)}
            for word, weight in explanation_list
        ]
