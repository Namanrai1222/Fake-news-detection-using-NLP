"""Check false-positive rates on full BBC news articles by topic."""

import json
import os
from zipfile import ZipFile

import pandas as pd

from src.config import BASE_DIR
from src.predict import Predictor
from src.preprocessing import clean_text


def main():
    archive = os.path.join(BASE_DIR, 'data', 'external', 'bbc-fulltext.zip')
    rows = []
    with ZipFile(archive) as file:
        for name in file.namelist():
            parts = name.split('/')
            if len(parts) == 3 and parts[0] == 'bbc' and parts[1] in {'business', 'entertainment', 'politics', 'sport', 'tech'} and name.endswith('.txt'):
                rows.append((parts[1], file.read(name).decode('latin-1')))
    data = pd.DataFrame(rows, columns=['topic', 'text'])
    data = data[data.text.str.split().str.len().between(20, 2000)].copy()
    predictor = Predictor()
    probabilities = predictor.calibrator.predict_proba(predictor.vectorizer.transform(data.text.map(clean_text)))[:, 1]
    data['predicted_fake'] = probabilities < .5
    result = {
        'sample_count': len(data),
        'flagged_fake_rate': round(float(data.predicted_fake.mean()), 4),
        'by_topic': {topic: {'sample_count': len(group), 'flagged_fake_rate': round(float(group.predicted_fake.mean()), 4)} for topic, group in data.groupby('topic')},
        'note': 'BBC articles are publisher news, not independently fact-checked truth labels.',
    }
    metrics_path = os.path.join(BASE_DIR, 'models', 'metrics_v2.json')
    with open(metrics_path) as file:
        report = json.load(file)
    report['general_news_full_articles'] = result
    with open(metrics_path, 'w') as file:
        json.dump(report, file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
