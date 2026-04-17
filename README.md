# Fake-news-detection-using-NLP

## Optional Ollama Explanation Layer

This project keeps the sklearn classifier as the primary prediction engine.
Ollama is optional and only used to generate a short natural-language explanation.

### Behavior

- Primary prediction: sklearn model (deterministic and fast).
- Optional explanation: Ollama local model.
- Safe fallback: if Ollama is unavailable, prediction still succeeds and response returns without LLM text.

### Recommended Small Models

- tinyllama
- llama3.2:1b
- qwen2.5:0.5b (if stable on your machine)

### Run Ollama (Optional)

1. Start Ollama server (separate terminal):
	- ollama serve
2. Pull a small model:
	- ollama pull tinyllama
	- or: ollama pull llama3.2:1b

### Enable in Backend

Set environment variables before starting Flask:

- OLLAMA_ENABLED=1
- OLLAMA_MODEL=tinyllama (or llama3.2:1b)
- OLLAMA_URL=http://127.0.0.1:11434/api/generate
- OLLAMA_TIMEOUT_SECONDS=8

PowerShell example:

1. $env:OLLAMA_ENABLED='1'
2. $env:OLLAMA_MODEL='tinyllama'
3. $env:OLLAMA_URL='http://127.0.0.1:11434/api/generate'
4. .\venv\Scripts\python.exe app\app.py

### API Response

The /predict response now includes:

- llm_explanation: optional string from Ollama
- llm_provider: "ollama" when explanation is returned

## Confidence Calibration Utility

To improve confidence reliability without retraining from scratch, run:

1. .\\venv\\Scripts\\python.exe -m src.calibrate_existing

This creates models/calibrator.pkl, which is loaded automatically by the backend.

## Security Hardening

The backend now includes safer defaults to reduce API/data exposure risks.

### Enabled by default

- Restricted CORS allowlist (localhost + 127.0.0.1 development origins)
- Request body size limit (MAX_REQUEST_BYTES)
- In-memory rate limiting (per client IP)
- Minimal `/health` output (no internal URLs/errors unless explicitly enabled)
- Structured API responses (`status: success|error`, `error` field)

### Optional production settings

Set these environment variables in production:

- APP_API_KEY: require `X-API-Key` for `/predict` and `/stats`
- ALLOWED_ORIGINS: comma-separated trusted frontend origins
- APP_HOST: use `127.0.0.1` by default; avoid `0.0.0.0` unless needed
- APP_DEBUG=0
- HEALTH_DETAILS=0

### Important

- Do NOT put API keys in frontend JavaScript or HTML.
- Keep `GOOGLE_FACTCHECK_API_KEY` and `APP_API_KEY` server-side only.
