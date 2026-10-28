"""Measure transfer to LIAR claims without treating mixed labels as binary truth."""

import json
import os

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from src.config import BASE_DIR
from src.predict import Predictor
from src.preprocessing import clean_text


def main():
    path = os.path.join(BASE_DIR, 'data', 'raw', 'test.tsv')
    data = pd.read_csv(path, sep='\t', header=None)
    data = data[data[1].isin(('true', 'mostly-true', 'false', 'pants-fire'))]
    actual = data[1].isin(('true', 'mostly-true')).astype(int)
    predictor = Predictor()
    vectors = predictor.vectorizer.transform(data[2].fillna('').map(clean_text))
    predicted = (predictor.calibrator.predict_proba(vectors)[:, 1] >= 0.5).astype(int)
    result = {
        'dataset': 'LIAR official test split, strict binary subset',
        'sample_count': len(data),
        'accuracy': round(accuracy_score(actual, predicted), 4),
        'f1_score': round(f1_score(actual, predicted), 4),
        'balanced_accuracy': round(balanced_accuracy_score(actual, predicted), 4),
        'confusion_matrix': confusion_matrix(actual, predicted).tolist(),
        'majority_baseline_accuracy': round(max(actual.mean(), 1 - actual.mean()), 4),
        'note': 'Out-of-domain political claims; not used in article model training or selection.',
    }
    metrics_path = os.path.join(BASE_DIR, 'models', 'metrics_v2.json')
    with open(metrics_path) as file:
        report = json.load(file)
    report['external_evaluation'] = result
    with open(metrics_path, 'w') as file:
        json.dump(report, file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
