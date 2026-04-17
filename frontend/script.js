const API_BASE = (window.location.port === '5500') ? 'http://127.0.0.1:5000' : '';
const LLM_TOGGLE_KEY = 'cipher.useLLM';
const ENSEMBLE_TOGGLE_KEY = 'cipher.useEnsemble';

async function apiFetch(path, options = {}) {
    return fetch(`${API_BASE}${path}`, options);
}

document.addEventListener('DOMContentLoaded', () => {
    const hasStatsWidgets = Boolean(document.getElementById('total-articles'));
    if (hasStatsWidgets) {
        fetchStats();
    }

    const analyzeBtn = document.getElementById('analyze-btn');
    const newsText = document.getElementById('news-text');
    const ctaBtn = document.getElementById('cta-btn');
    const llmToggle = document.getElementById('llm-toggle');
    const ensembleToggle = document.getElementById('ensemble-toggle');

    if (llmToggle) {
        const saved = localStorage.getItem(LLM_TOGGLE_KEY);
        llmToggle.checked = saved === null ? true : saved === 'true';
        llmToggle.addEventListener('change', () => {
            localStorage.setItem(LLM_TOGGLE_KEY, String(llmToggle.checked));
        });
    }

    if (ensembleToggle) {
        const saved = localStorage.getItem(ENSEMBLE_TOGGLE_KEY);
        ensembleToggle.checked = saved === null ? true : saved === 'true';
        ensembleToggle.addEventListener('change', () => {
            localStorage.setItem(ENSEMBLE_TOGGLE_KEY, String(ensembleToggle.checked));
        });
    }

    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', handleAnalysis);
    }

    if (ctaBtn) {
        ctaBtn.addEventListener('click', () => {
            const demoTarget = document.getElementById('demo-target');
            if (demoTarget) {
                demoTarget.scrollIntoView({ behavior: 'smooth', block: 'start' });
                newsText?.focus();
                return;
            }
            window.location.href = 'analyzer.html';
        });
    }
    
    // Ctrl+Enter shortcut
    if (newsText) {
        newsText.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                handleAnalysis();
            }
        });
    }
});

const escapeHtml = (unsafe) => {
    return String(unsafe)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
};

async function handleAnalysis() {
    const textBase = document.getElementById('news-text')?.value.trim() || '';
    const llmToggle = document.getElementById('llm-toggle');
    const ensembleToggle = document.getElementById('ensemble-toggle');
    const useLLM = llmToggle ? llmToggle.checked : true;
    const useEnsemble = ensembleToggle ? ensembleToggle.checked : true;
    if (!textBase) {
        alert('Please enter news text before running analysis.');
        return;
    }

    const loader = document.getElementById('loader');
    const resultsPane = document.getElementById('results-target');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingMsg = document.getElementById('loading-msg');

    if (!loader || !resultsPane || !analyzeBtn || !loadingMsg) {
        return;
    }

    // UI Reset
    analyzeBtn.disabled = true;
    loader.classList.remove('hidden');
    resultsPane.classList.add('hidden');
    
    // Smooth reset of bars
    document.getElementById('conf-prog').style.width = '0%';
    document.getElementById('conf-pct').textContent = '--%';

    try {
        let response = await apiFetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: textBase, use_llm: useLLM, use_ensemble: useEnsemble })
        });

        let data = await response.json().catch(() => ({}));

        if (response.ok && data.status === 'error') {
            response = { ok: false, status: 400 };
        }

        if (response.status === 503 && String(data.error || '').toLowerCase().includes('initializing')) {
            loadingMsg.textContent = 'Model is warming up. Retrying shortly...';
            const isReady = await waitForModelReady(8, 1200);
            if (isReady) {
                response = await apiFetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: textBase, use_llm: useLLM, use_ensemble: useEnsemble })
                });
                data = await response.json().catch(() => ({}));
            }
        }

        if (!response.ok) {
            loadingMsg.textContent = "Analysis Interrupted";
            loadingMsg.style.color = "var(--neon-red)";
            alert(`Backend log: ${data.error || 'Unknown error occurred.'}`);
            setTimeout(() => {
                loadingMsg.textContent = "Synthesizing matrices...";
                loadingMsg.style.color = "var(--neon-purple)";
                loader.classList.add('hidden');
            }, 3000);
            return;
        }

        renderResults(data || {});
    } catch (err) {
        alert('The backend engine is unreachable. Check if the Flask server is running.');
    } finally {
        analyzeBtn.disabled = false;
        loader.classList.add('hidden');
    }
}

async function waitForModelReady(maxAttempts, delayMs) {
    for (let i = 0; i < maxAttempts; i += 1) {
        try {
            const response = await apiFetch('/health');
            const data = await response.json().catch(() => ({}));
            if (response.ok && data.model_loaded === true) {
                return true;
            }
        } catch (err) {
            // Ignore transient health-check failures and keep retrying.
        }
        await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
    return false;
}

function renderResults(data) {
    const resultsPane = document.getElementById('results-target');
    const badge = document.getElementById('pred-badge');
    const confFill = document.getElementById('conf-prog');
    const confText = document.getElementById('conf-pct');
    const confNote = document.getElementById('conf-note');
    const biasRisk = document.getElementById('bias-risk');
    const reliabilityBand = document.getElementById('reliability-band');
    const limeList = document.getElementById('lime-explanation');
    const reasoningList = document.getElementById('reasoning-list');
    const verificationStatus = document.getElementById('verification-status');
    const verificationDetails = document.getElementById('verification-details');
    const llmExplanationEl = document.getElementById('llm-explanation');
    const biasList = document.getElementById('bias-hints');

    if (!resultsPane || !badge || !confFill || !confText || !biasRisk || !limeList || !biasList) {
        return;
    }

    // Build Badges
    const prediction = data.prediction === 'Fake' ? 'Fake' : 'Real';
    badge.className = 'result-badge ' + (prediction === 'Fake' ? 'badge-fake' : 'badge-real');
    badge.textContent = prediction;

    // Build Confidence Bar
    const confidence = Number(data.confidence_calibrated ?? data.confidence);
    const rawConfidence = Number(data.confidence_raw ?? data.confidence);
    const confVal = Number.isFinite(confidence) ? (confidence * 100).toFixed(1) : '0.0';
    confText.textContent = `${confVal}%`;
    if (confNote) {
        const rawPct = Number.isFinite(rawConfidence) ? (rawConfidence * 100).toFixed(1) : null;
        const ensembleFlag = data.ensemble_requested === true ? ' Ensemble enabled.' : ' Ensemble disabled.';
        confNote.textContent = rawPct ? `Calibrated from raw ${rawPct}%.${ensembleFlag}` : `Calibrated confidence shown.${ensembleFlag}`;
    }
    
    // Color code bar based on certainty
    let barColor = 'linear-gradient(90deg, var(--neon-blue), var(--neon-purple))';
    if (prediction === 'Fake') {
        barColor = 'linear-gradient(90deg, #f43f5e, #9f1239)';
    } else {
        barColor = 'linear-gradient(90deg, #10b981, #047857)';
    }
    confFill.style.background = barColor;

    // Force reflow for animation
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            confFill.style.width = `${confVal}%`;
        });
    });

    // Bias mapping
    biasRisk.textContent = data.bias_analysis?.risk || 'Unknown';
    biasRisk.style.color = (biasRisk.textContent === "High") ? "var(--neon-red)" : "inherit";

    if (reliabilityBand) {
        reliabilityBand.textContent = data.reliability?.band || 'Unknown';
        if (reliabilityBand.textContent === 'High') {
            reliabilityBand.style.color = 'var(--neon-green)';
        } else if (reliabilityBand.textContent === 'Medium') {
            reliabilityBand.style.color = 'var(--neon-blue)';
        } else {
            reliabilityBand.style.color = 'var(--neon-red)';
        }
    }

    // Feature signal output
    limeList.innerHTML = '';
    if (data.feature_signals && data.feature_signals.length > 0) {
        data.feature_signals.forEach(item => {
            const li = document.createElement('li');
            const impact = Number(item.impact ?? 0);
            const direction = escapeHtml(item.direction || 'Neutral');
            li.innerHTML = `<span>"${escapeHtml(item.term || item.word || 'token')}"</span> <span>${impact.toFixed(4)} (${direction})</span>`;
            limeList.appendChild(li);
        });
    } else if (data.explanation && data.explanation.length > 0) {
        data.explanation.forEach(item => {
            const li = document.createElement('li');
            const impact = Number(item.impact ?? item.weight ?? 0);
            li.innerHTML = `<span>"${escapeHtml(item.word)}"</span> <span>${impact.toFixed(4)}</span>`;
            limeList.appendChild(li);
        });
    } else {
        limeList.innerHTML = '<li><span>No strong local feature signal detected.</span></li>';
    }

    if (reasoningList) {
        reasoningList.innerHTML = '';
        const reasons = Array.isArray(data.reasoning) ? data.reasoning : [];
        if (reasons.length > 0) {
            reasons.forEach((reason) => {
                const li = document.createElement('li');
                li.textContent = reason;
                reasoningList.appendChild(li);
            });
        } else {
            reasoningList.innerHTML = '<li>Reasoning not available.</li>';
        }
    }

    if (verificationStatus && verificationDetails) {
        const v = data.verification || {};
        verificationStatus.textContent = `Status: ${v.status || 'not_available'}`;
        if (typeof v.details === 'string') {
            verificationDetails.textContent = v.details;
        } else if (v.details && typeof v.details === 'object') {
            const publisher = v.details.publisher || 'unknown publisher';
            const rating = v.details.rating || 'no rating';
            verificationDetails.textContent = `${publisher}, ${rating}`;
        } else {
            verificationDetails.textContent = 'No verification details available.';
        }
    }

    if (llmExplanationEl) {
        if (data.llm_explanation) {
            llmExplanationEl.textContent = data.llm_explanation;
        } else if (data.llm_requested === false) {
            llmExplanationEl.textContent = 'LLM explanation is disabled for this request.';
        } else {
            llmExplanationEl.textContent = 'Optional Ollama explanation not available.';
        }
    }

    // Bias Hints Output
    biasList.innerHTML = '';
    if (data.bias_analysis?.indicators && data.bias_analysis.indicators.length > 0) {
        data.bias_analysis.indicators.forEach(hint => {
            const li = document.createElement('li');
            li.textContent = hint;
            biasList.appendChild(li);
        });
    } else {
        biasList.innerHTML = '<li>Clean taxonomy detected (No targeted sources).</li>';
    }

    // Unhide Results
    resultsPane.classList.remove('hidden');
    resultsPane.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function fetchStats() {
    const totalArticlesEl = document.getElementById('total-articles');
    const accuracyEl = document.getElementById('val-accuracy');
    const f1El = document.getElementById('val-f1');
    const eceEl = document.getElementById('val-ece');
    const brierEl = document.getElementById('val-brier');
    const calibrationSummary = document.getElementById('calibration-summary');
    const eceText = document.getElementById('ece-text');
    const brierText = document.getElementById('brier-text');
    const eceBar = document.getElementById('ece-bar');
    const brierBar = document.getElementById('brier-bar');

    if (!totalArticlesEl || !accuracyEl || !f1El) {
        return;
    }

    try {
        const response = await apiFetch('/stats');
        if (!response.ok) {
            throw new Error(`Stats API returned ${response.status}`);
        }
        const data = await response.json();
        if (data.status === 'error') {
            throw new Error(data.error || 'Stats API returned an error status');
        }
        
        const totalArticles = Number(data.total_articles || 0);
        totalArticlesEl.textContent = totalArticles.toLocaleString();
        
        if (data.model_metrics) {
            const accuracy = Number(data.model_metrics.accuracy || 0);
            const f1 = Number(data.model_metrics.f1_score || 0);
            const ece = Number(data.model_metrics.ece);
            const brier = Number(data.model_metrics.brier_score);
            const acc = (accuracy * 100).toFixed(1);
            accuracyEl.textContent = `${acc}%`;
            
            f1El.textContent = f1.toFixed(3);

            if (eceEl) {
                eceEl.textContent = Number.isFinite(ece) ? ece.toFixed(3) : '--';
            }
            if (brierEl) {
                brierEl.textContent = Number.isFinite(brier) ? brier.toFixed(3) : '--';
            }

            if (eceText) {
                eceText.textContent = Number.isFinite(ece) ? ece.toFixed(3) : '--';
            }
            if (brierText) {
                brierText.textContent = Number.isFinite(brier) ? brier.toFixed(3) : '--';
            }

            if (eceBar && Number.isFinite(ece)) {
                const pct = Math.max(0, Math.min(100, ece * 100));
                eceBar.style.width = `${pct}%`;
            }
            if (brierBar && Number.isFinite(brier)) {
                const pct = Math.max(0, Math.min(100, brier * 100));
                brierBar.style.width = `${pct}%`;
            }

            if (calibrationSummary) {
                if (Number.isFinite(ece) && Number.isFinite(brier)) {
                    let label = 'Moderate calibration quality';
                    if (ece <= 0.05 && brier <= 0.05) {
                        label = 'Strong calibration quality';
                    } else if (ece >= 0.15 || brier >= 0.10) {
                        label = 'Weak calibration quality';
                    }
                    calibrationSummary.textContent = `${label} (ECE ${ece.toFixed(3)}, Brier ${brier.toFixed(3)}).`;
                } else {
                    calibrationSummary.textContent = 'Calibration metrics unavailable in current model artifact.';
                }
            }
            
            // Build Matrix
            const cm = data.model_metrics.confusion_matrix;
            if (cm && cm.length === 2) {
                const cm00 = document.getElementById('cm-00');
                const cm01 = document.getElementById('cm-01');
                const cm10 = document.getElementById('cm-10');
                const cm11 = document.getElementById('cm-11');
                if (cm00 && cm01 && cm10 && cm11) {
                    cm00.textContent = cm[0][0];
                    cm01.textContent = cm[0][1];
                    cm10.textContent = cm[1][0];
                    cm11.textContent = cm[1][1];
                }
            }
        }
    } catch (err) {
        console.error("Failed fetching model metrics: ", err);
    }
}
