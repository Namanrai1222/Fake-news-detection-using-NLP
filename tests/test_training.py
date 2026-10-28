import pandas as pd

from main import prepare, split_articles


def test_prepare_removes_conflicting_labels_and_duplicate_text():
    text = 'A report about an ordinary city event with enough detail to qualify as a news article. ' * 3
    other = 'A different report about a business event with enough detail to qualify as a news article. ' * 3
    data = pd.DataFrame({'text': [text, text, other, other], 'label': [0, 1, 1, 1], 'title': ['one', 'one', 'two', 'two']})
    prepared = prepare(data)
    assert len(prepared) == 1
    assert prepared.iloc[0].label == 1


def test_split_keeps_matching_headlines_together():
    rows = []
    for group in range(30):
        headline = f'News headline {chr(97 + group // 26)}{chr(97 + group % 26)}'
        for version in range(2):
            rows.append({'title': headline, 'text': f'Article {group} version {version}', 'cleaned': f'article {group} version {version}', 'label': group % 2})
    parts = split_articles(pd.DataFrame(rows), 42)
    headline_sets = [set(part.title) for part in parts]
    assert sum(len(part) for part in parts) == 60
    assert all(not headline_sets[i] & headline_sets[j] for i in range(4) for j in range(i + 1, 4))
