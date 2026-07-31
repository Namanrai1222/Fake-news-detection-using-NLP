# Fake News Detection using NLP

A machine learning system that classifies news articles as real or fake using NLP feature extraction and a scikit-learn classifier, with an optional local LLM layer for natural-language explanations.

[![Python](https://img.shields.io/badge/Python-54.6%25-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask-black)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)](https://scikit-learn.org/)

## Overview

This project detects misinformation in news text using a classical NLP + machine learning pipeline. It's designed with a clear separation between the deterministic ML model (fast, reproducible) and an optional generative layer (Ollama) that explains predictions in plain English - the system never depends on the LLM to function.

## Key Features

- **Primary prediction engine**: scikit-learn classifier trained on news text features - deterministic and fast
- **Optional AI explanation layer**: local Ollama models (tinyllama, llama3.2:1b, qwen2.5:0.5b) generate a short natural-language rationale for each prediction, with automatic safe fallback if Ollama is unavailable
- **Confidence calibration**: a dedicated calibration utility (`src/calibrate_existing`) improves confidence score reliability without full retraining
- **Bias checking**: includes a `bias_check.py` utility to evaluate model fairness across inputs
- **Explainability**: LIME-based debugging (`debug_lime.py`) for inspecting model decisions
- **Production-hardened API**: restricted CORS, request size limits, per-IP rate limiting, minimal error leakage on `/health`, and optional API-key auth

## Tech Stack

| Layer | Technology |
|---|---|
| ML / Data | Python, scikit-learn, pandas, NumPy |
| Backend API | Flask |
| Frontend | HTML, CSS, JavaScript |
| Optional LLM | Ollama (local inference, e.g. tinyllama) |
| Explainability | LIME |

## Architecture

- `app/` - Flask backend serving the `/predict` and `/stats` endpoints
- `src/` - core ML pipeline, including the confidence calibrator
- `models/` - trained classifier and calibrator artifacts
- `notebooks/` - exploratory data analysis and model development
- `frontend/` - client UI for submitting articles and viewing predictions
- `tests/` - test suite

## Security Hardening

Since this exposes a public-facing prediction API, it includes production safeguards by default:

- Restricted CORS allowlist (localhost + 127.0.0.1 development origins)
- Request body size limit (`MAX_REQUEST_BYTES`)
- In-memory per-IP rate limiting
- Minimal `/health` output (no internal URLs/errors unless explicitly enabled)
- Structured API responses (`status: success|error`)
- Secrets (`APP_API_KEY`, `GOOGLE_FACTCHECK_API_KEY`) are server-side only, never exposed to frontend code

### Optional production settings

- `APP_API_KEY`: require `X-API-Key` header for `/predict` and `/stats`
- `ALLOWED_ORIGINS`: comma-separated trusted frontend origins
- `APP_HOST`: use `127.0.0.1` by default; avoid `0.0.0.0` unless needed
- `APP_DEBUG=0`
- `HEALTH_DETAILS=0`

## Getting Started

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python app/app.py
```

### Optional: enable the LLM explanation layer

```powershell
$env:OLLAMA_ENABLED='1'
$env:OLLAMA_MODEL='tinyllama'
$env:OLLAMA_URL='http://127.0.0.1:11434/api/generate'
ollama serve
ollama pull tinyllama
```

### Optional: improve confidence calibration

```bash
python -m src.calibrate_existing
```

This creates `models/calibrator.pkl`, loaded automatically by the backend.

## API Response

`/predict` returns a structured JSON response including the classification and confidence score. When Ollama is enabled, the response also includes:

- `llm_explanation`: optional natural-language rationale
- `llm_provider`: `"ollama"` when an explanation is returned

## What I Learned

Building this project involved balancing model interpretability with production safety - keeping the core classifier deterministic while layering in an optional LLM explanation without introducing a hard dependency, plus hardening a public ML API against common exposure risks (CORS misconfiguration, unbounded request size, missing rate limits).

## License

MIT
