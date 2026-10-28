# Cipher Sentinel

An English news article classifier with a Flask API and a static frontend. It estimates whether article language resembles examples labeled real or fake in research datasets. **It cannot establish that a claim is true or false.**

## Run locally

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python app/app.py
```

Open `http://127.0.0.1:5000/`. The repository includes trained artifacts, so retraining is optional. Load `joblib` model files only from a trusted repository because deserialization can execute code.

## Reproduce the training and evaluations

```bash
pip install -r requirements-train.txt
python -m src.download_datasets
python main.py
python -m src.evaluate_bbc
python -m src.evaluate_liar
python -m pytest -q
```

The download script pins each remote dataset revision and checks SHA-256. Files are saved under ignored `data/external/` and are not committed. Training uses bundled [ISOT](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/) articles and unique articles from [WELFake](https://zenodo.org/records/4561253). The WELFake Parquet mirror's numeric labels are reversed relative to overlapping ISOT articles; training checks 37,653 matching cleaned articles, reverses the labels only when more than 99% disagree, and excludes every overlap. This adds 22,167 unique articles, for 59,820 total after filtering.

The script removes empty, short, duplicate, and conflicting examples. It makes separate article training, validation, calibration, and test sets, grouped by headline **within each source** to reduce near-duplicate leakage. Four logistic regression candidates vary TF–IDF vocabulary size, regularization, and the number of [AG News](https://huggingface.co/datasets/fancyzhx/ag_news/blob/main/README.md) examples. AG News has **topic labels, not fact-check labels**; selected publisher-news snippets serve as weak examples of ordinary real news. Candidate selection keeps models within one F1 point of the best article validation score, then chooses the lowest AG News validation false-positive rate. Sigmoid calibration uses only the article calibration set. The official AG News test split, [BBC](http://mlg.ucd.ie/datasets/bbc.html) full articles, and [LIAR](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) test claims are excluded from fitting and selection. Split and fitting seeds are fixed in `main.py` and `src/train.py` for reproducibility.

## Measured results

| Evaluation | Size | Result |
|---|---:|---:|
| Held-out ISOT + new WELFake articles | 5,998 | Accuracy 94.08%; F1 0.9461 |
| ISOT article subset | 3,765 | F1 0.9769 |
| New WELFake article subset | 2,233 | F1 0.8935 |
| AG News official test snippets | 6,665 | 0.86% flagged fake |
| BBC independent full articles | 2,218 | 2.34% flagged fake |
| LIAR short political claims | 790 | Accuracy 57.59%; balanced accuracy 52.08% |

AG News and BBC are publisher-news stress tests, **not verified truth benchmarks**. The BBC tech subset has the highest false-positive rate at 7.25%. LIAR's majority-class accuracy is 56.84%; the small gain and 52.08% balanced accuracy show weak transfer to short claims. Same-collection article scores can still reflect source and era shortcuts.

Training F1 on article examples is 0.9650 and article validation F1 is 0.9467, a 1.83 percentage-point gap. With 25%, 50%, and 100% of the training examples, validation F1 rises from 0.9211 to 0.9371 to 0.9467. This argues against a severe underfit, while the remaining gap and lower WELFake score indicate some overfit and domain shift. Relative to the best article-only candidate, the selected ordinary-news-augmented model lowers article validation F1 from 0.9531 to 0.9467 and WELFake validation F1 from 0.9139 to 0.8912, while lowering AG News validation false positives from 55.15% to 0.80%. This is a deliberate trade-off for ordinary news coverage. Full candidate scores, split counts, confusion matrices, calibration results, and topic rates are in `models/metrics_v2.json` and on the Evidence page.

## API

`POST /predict` accepts JSON `{ "text": "...", "use_llm": false }` with 20–2,500 words of English article text. It returns `prediction`, `confidence_calibrated`, `feature_signals`, optional `verification`, and optional `llm_explanation`. `GET /stats` returns saved evaluation results; `GET /health` reports model readiness. Requests can fail with 400 (invalid text), 429 (rate limit), or 503 (model unavailable).

Optional Google Fact Check lookup requires `GOOGLE_FACTCHECK_API_KEY`; no match is not a truth verdict. Optional local Ollama explanations require `OLLAMA_ENABLED=1` and a running Ollama server. LLM text is explanatory only. See `.env.example` for deployment settings including `APP_API_KEY`, `ALLOWED_ORIGINS`, and `APP_HOST`. A browser-only frontend cannot keep an API key secret. Set HTTPS at the reverse proxy. Rate limits are in memory and scoped to one Flask process.

## Limits

The model was trained on historical English text. It has not been validated for current events, other languages, satire, or standalone claim verification. Check original reporting and independent sources for consequential decisions.
