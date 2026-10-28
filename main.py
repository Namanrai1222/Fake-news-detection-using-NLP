"""Train on deduplicated articles and audit ordinary-news false positives."""

import json
import os
from datetime import date

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from src.config import BASE_DIR, FAKE_DATA_PATH, TRUE_DATA_PATH, MODEL_PATH, VECTORIZER_PATH
from src.data_loader import load_data
from src.evaluate import expected_calibration_error
from src.preprocessing import clean_text
from src.train import save_model, train_model

EXTERNAL = os.path.join(BASE_DIR, 'data', 'external')
TOPICS = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}


def prepare(data):
    data = data.dropna(subset=['text']).copy()
    data = data[data['text'].str.split().str.len().between(20, 2000)].copy()
    data['cleaned'] = data['text'].map(clean_text)
    data = data[data['cleaned'].str.split().str.len() >= 20].copy()
    conflicts = data.groupby('cleaned')['label'].nunique()
    data = data[~data['cleaned'].isin(conflicts[conflicts > 1].index)]
    return data.drop_duplicates('cleaned').reset_index(drop=True)


def split_articles(data, seed):
    """Keep identical headlines in one split to reduce near-duplicate leakage."""
    groups = data['title'].fillna(data['text'].str[:100]).map(clean_text)
    groups = groups.where(groups.ne(''), data['cleaned'].str[:100])
    def split(frame, group, fraction, random_state):
        left, right = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=random_state).split(frame, groups=group))
        return frame.iloc[left].copy(), frame.iloc[right].copy(), group.iloc[left], group.iloc[right]
    train, hold, train_group, hold_group = split(data, groups, .30, seed)
    valid, rest, _, rest_group = split(hold, hold_group, 2/3, seed + 1)
    calibrate, test, _, _ = split(rest, rest_group, .5, seed + 2)
    return train, valid, calibrate, test


def load_sources():
    isot = prepare(load_data(FAKE_DATA_PATH, TRUE_DATA_PATH))
    isot['source'] = 'ISOT'
    welfake_path = os.path.join(EXTERNAL, 'welfake.parquet')
    if not os.path.exists(welfake_path):
        raise FileNotFoundError('WELFake missing. Run python -m src.download_datasets first.')
    welfake = prepare(pd.read_parquet(welfake_path))
    # The hosted WELFake numeric labels disagree with overlapping ISOT articles.
    # Correct them only if this anchor still holds.
    overlap = welfake[['cleaned', 'label']].merge(isot[['cleaned', 'label']], on='cleaned', suffixes=('_wel', '_isot'))
    inverse_rate = (overlap.label_wel.ne(overlap.label_isot)).mean()
    if len(overlap) < 1000 or inverse_rate < .99:
        raise ValueError('WELFake label mapping changed; inspect source before training.')
    welfake['label'] = 1 - welfake['label']
    welfake = welfake[~welfake.cleaned.isin(isot.cleaned)].copy()
    welfake['source'] = 'WELFake (nonoverlap)'
    print(f'Unique articles: ISOT={len(isot)}, new WELFake={len(welfake)}; reversed-label anchor={len(overlap)} ({inverse_rate:.4%})')
    return isot, welfake, len(overlap)


def load_general_news(excluded):
    train = pd.read_parquet(os.path.join(EXTERNAL, 'ag_news_train.parquet'))
    test = pd.read_parquet(os.path.join(EXTERNAL, 'ag_news_test.parquet'))
    def clean(frame):
        frame = frame[frame.text.str.split().str.len() >= 20].copy()
        frame['cleaned'] = frame.text.map(clean_text)
        return frame[frame.cleaned.str.split().str.len() >= 20].drop_duplicates('cleaned')
    test = clean(test)
    train = clean(train)
    train = train[~train.cleaned.isin(set(excluded) | set(test.cleaned))]
    test = test[~test.cleaned.isin(excluded)]
    # AG News has topic labels, not truth ratings. These are weak real-news examples.
    sampled = train.groupby('label', group_keys=False).sample(n=5000, random_state=42)
    sampled_train, sampled_valid, _, _ = split_articles(sampled.assign(title=sampled.text.str[:100]), 51)
    return sampled_train.assign(topic=sampled_train.label, label=1), sampled_valid.assign(topic=sampled_valid.label), test


def scores(model, vectorizer, frame):
    pred = model.predict(vectorizer.transform(frame.cleaned))
    return float(f1_score(frame.label, pred))


def report_metrics(calibrator, vectorizer, frame):
    prob = calibrator.predict_proba(vectorizer.transform(frame.cleaned))[:, 1]
    pred = (prob >= .5).astype(int)
    actual = frame.label.to_numpy()
    return {
        'sample_count': len(frame),
        'accuracy': round(accuracy_score(actual, pred), 4),
        'precision': round(precision_score(actual, pred), 4),
        'recall': round(recall_score(actual, pred), 4),
        'f1_score': round(f1_score(actual, pred), 4),
        'brier_score': round(brier_score_loss(actual, prob), 4),
        'ece': round(expected_calibration_error(actual, prob), 4),
        'confusion_matrix': confusion_matrix(actual, pred).tolist(),
    }


def main():
    isot, welfake, anchor_count = load_sources()
    i_train, i_valid, i_cal, i_test = split_articles(isot, 42)
    w_train, w_valid, w_cal, w_test = split_articles(welfake, 142)
    article_train = pd.concat([i_train, w_train], ignore_index=True)
    article_valid = pd.concat([i_valid, w_valid], ignore_index=True)
    article_cal = pd.concat([i_cal, w_cal], ignore_index=True)
    article_test = pd.concat([i_test, w_test], ignore_index=True)
    parts = [article_train, article_valid, article_cal, article_test]
    assert sum(len(part) for part in parts) == len(set(pd.concat([part.cleaned for part in parts])))
    ag_train, ag_valid, ag_test = load_general_news(pd.concat([isot.cleaned, welfake.cleaned]))
    print(f'Splits: articles train={len(article_train)} valid={len(article_valid)} calibration={len(article_cal)} test={len(article_test)}; AG train={len(ag_train)} valid={len(ag_valid)} test={len(ag_test)}')

    candidates = []
    for ag_count, features, c in ((0, 5000, 1.0), (6000, 5000, 1.0), (12000, 40000, 1.0), (12000, 40000, .3)):
        extra = ag_train.groupby('topic', group_keys=False).sample(n=ag_count // 4, random_state=42) if ag_count else ag_train.iloc[:0]
        fit = pd.concat([article_train[['cleaned', 'label']], extra[['cleaned', 'label']]], ignore_index=True)
        model, vectorizer = train_model(fit.cleaned, fit.label, max_features=features, c=c)
        valid_f1 = scores(model, vectorizer, article_valid)
        source_f1 = {source: scores(model, vectorizer, frame) for source, frame in [('ISOT', i_valid), ('WELFake', w_valid)]}
        ag_fpr = float((model.predict(vectorizer.transform(ag_valid.cleaned)) == 0).mean())
        train_f1 = scores(model, vectorizer, article_train)
        row = {'ag_examples': len(extra), 'features': features, 'C': c, 'train_f1': round(train_f1, 4), 'validation_f1': round(valid_f1, 4), 'validation_source_f1': {k: round(v, 4) for k, v in source_f1.items()}, 'validation_general_news_false_positive_rate': round(ag_fpr, 4)}
        candidates.append(row)
        print('Candidate', row, flush=True)
    best_article_f1 = max(row['validation_f1'] for row in candidates)
    eligible = [row for row in candidates if row['validation_f1'] >= best_article_f1 - .01]
    selected = min(eligible, key=lambda row: (row['validation_general_news_false_positive_rate'], -row['validation_f1']))
    extra = ag_train.groupby('topic', group_keys=False).sample(n=selected['ag_examples'] // 4, random_state=42) if selected['ag_examples'] else ag_train.iloc[:0]
    fit = pd.concat([article_train[['cleaned', 'label']], extra[['cleaned', 'label']]], ignore_index=True)
    model, vectorizer = train_model(fit.cleaned, fit.label, selected['features'], selected['C'])
    calibrator = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid')
    calibrator.fit(vectorizer.transform(article_cal.cleaned), article_cal.label)
    save_model(model, vectorizer, MODEL_PATH, VECTORIZER_PATH)
    joblib.dump(calibrator, os.path.join(BASE_DIR, 'models', 'calibrator.pkl'))

    # Learning curve: rising validation F1 with more data argues against underfitting;
    # a widening train/validation gap is evidence of overfitting.
    learning_curve = []
    for fraction in (.25, .5, 1.0):
        subset_articles = article_train.sample(frac=fraction, random_state=42)
        subset_extra = extra.sample(frac=fraction, random_state=42) if len(extra) else extra
        subset = pd.concat([subset_articles[['cleaned', 'label']], subset_extra[['cleaned', 'label']]], ignore_index=True)
        trial, trial_vectorizer = train_model(subset.cleaned, subset.label, selected['features'], selected['C'])
        learning_curve.append({'training_fraction': fraction, 'train_article_f1': round(scores(trial, trial_vectorizer, subset_articles), 4), 'validation_article_f1': round(scores(trial, trial_vectorizer, article_valid), 4)})
    ag_prob = calibrator.predict_proba(vectorizer.transform(ag_test.cleaned))[:, 1]
    ag_test = ag_test.assign(predicted_fake=ag_prob < .5)
    general_news = {
        'sample_count': len(ag_test),
        'false_positive_rate': round(float(ag_test.predicted_fake.mean()), 4),
        'by_topic': {TOPICS[k]: {'sample_count': len(group), 'false_positive_rate': round(float(group.predicted_fake.mean()), 4)} for k, group in ag_test.groupby('label')},
        'note': 'AG News has topic labels, not verified truth labels; this is a publisher-news stress test.',
    }
    report = {
        'model_metrics': report_metrics(calibrator, vectorizer, article_test),
        'source_metrics': {'ISOT': report_metrics(calibrator, vectorizer, i_test), 'WELFake_nonoverlap': report_metrics(calibrator, vectorizer, w_test)},
        'general_news': general_news,
        'dataset': 'ISOT + nonoverlapping WELFake articles; optional AG News weak real-news augmentation',
        'total_articles': len(isot) + len(welfake),
        'split_sizes': {'train': len(article_train), 'validation': len(article_valid), 'calibration': len(article_cal), 'test': len(article_test)},
        'selected_model': selected,
        'candidates': candidates,
        'learning_curve': learning_curve,
        'welfake_reversed_label_anchor_count': anchor_count,
        'training_date': date.today().isoformat(),
        'evaluation_note': 'Article holdouts are grouped by headline; AG News false positives are a weak-label stress test, not fact verification.',
    }
    with open(os.path.join(BASE_DIR, 'models', 'metrics_v2.json'), 'w') as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
