const api = (path, options) => fetch(path, options);

const textInput = document.getElementById('news-text');
if (textInput) {
  const count = document.getElementById('word-count');
  textInput.addEventListener('input', () => {
    const words = textInput.value.trim().split(/\s+/).filter(Boolean).length;
    count.textContent = `${words.toLocaleString()} word${words === 1 ? '' : 's'}`;
  });
  textInput.addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') analyze();
  });
  document.getElementById('analyze-btn').addEventListener('click', analyze);
}

async function analyze() {
  const text = textInput.value.trim();
  const words = text.split(/\s+/).filter(Boolean).length;
  const error = document.getElementById('input-error');
  const status = document.getElementById('loading-msg');
  const button = document.getElementById('analyze-btn');
  error.hidden = true;
  if (words < 20 || words > 2500) {
    error.textContent = 'Please enter between 20 and 2,500 words of article text.';
    error.hidden = false;
    textInput.focus();
    return;
  }
  button.disabled = true;
  status.textContent = 'Analyzing article…';
  try {
    const response = await api('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, use_llm: document.getElementById('llm-toggle').checked}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Analysis is unavailable.');
    renderResult(data);
    status.textContent = 'Analysis complete.';
  } catch (cause) {
    error.textContent = cause.message === 'Failed to fetch' ? 'Cannot reach the server. Start the Flask app and try again.' : cause.message;
    error.hidden = false;
    status.textContent = '';
  } finally {
    button.disabled = false;
  }
}

function renderResult(data) {
  const panel = document.getElementById('results-target');
  const badge = document.getElementById('pred-badge');
  badge.textContent = data.prediction === 'Fake' ? 'Resembles fake' : 'Resembles real';
  badge.classList.toggle('is-fake', data.prediction === 'Fake');
  const confidence = Math.max(0, Math.min(1, Number(data.confidence_calibrated) || 0));
  document.getElementById('conf-pct').textContent = `${(confidence * 100).toFixed(1)}%`;
  document.getElementById('conf-prog').style.width = `${confidence * 100}%`;
  const signals = document.getElementById('lime-explanation');
  signals.replaceChildren();
  const items = Array.isArray(data.feature_signals) ? data.feature_signals.slice(0, 6) : [];
  if (!items.length) {
    const item = document.createElement('li'); item.textContent = 'No prominent terms found.'; signals.append(item);
  }
  for (const signal of items) {
    const item = document.createElement('li');
    const term = document.createElement('span'); term.textContent = signal.term || 'Term';
    const direction = document.createElement('span'); direction.textContent = `Toward ${signal.direction === 'Fake' ? 'fake' : 'real'}`;
    item.append(term, direction); signals.append(item);
  }
  const verification = data.verification || {};
  const verificationStatus = document.getElementById('verification-status');
  const verificationDetails = document.getElementById('verification-details');
  if (verification.status === 'verified_match_found') {
    verificationStatus.textContent = 'A matching published fact check was found.';
    const detail = verification.details || {};
    verificationDetails.textContent = [detail.publisher, detail.rating].filter(Boolean).join(' / ') || 'Review the matched fact check independently.';
  } else {
    verificationStatus.textContent = 'No external fact-check result for this analysis.';
    verificationDetails.textContent = verification.status === 'not_configured' ? 'The server has no fact-check API key configured.' : 'A missing match does not indicate that a claim is true or false.';
  }
  document.getElementById('llm-explanation').textContent = data.llm_explanation || (data.llm_requested ? 'Local Ollama explanation unavailable.' : 'Not requested.');
  panel.hidden = false;
  panel.scrollIntoView({behavior: 'smooth', block: 'start'});
}

if (document.getElementById('val-accuracy')) loadStats();
async function loadStats() {
  try {
    const response = await api('/stats');
    const report = await response.json();
    if (!response.ok) throw new Error(report.error || 'Metrics unavailable.');
    const m = report.model_metrics || {};
    const ext = report.external_evaluation || {};
    const source = report.source_metrics || {};
    const ag = report.general_news || {};
    const bbc = report.general_news_full_articles || {};
    const pct = (value) => typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—';
    const num = (value) => typeof value === 'number' ? value.toLocaleString() : '—';
    document.getElementById('val-accuracy').textContent = pct(m.accuracy);
    document.getElementById('val-f1').textContent = typeof m.f1_score === 'number' ? m.f1_score.toFixed(3) : '—';
    document.getElementById('total-articles').textContent = num(report.total_articles);
    document.getElementById('val-ece').textContent = typeof m.ece === 'number' ? m.ece.toFixed(3) : '—';
    document.getElementById('isot-f1').textContent = typeof source.ISOT?.f1_score === 'number' ? source.ISOT.f1_score.toFixed(3) : '—';
    document.getElementById('welfake-f1').textContent = typeof source.WELFake_nonoverlap?.f1_score === 'number' ? source.WELFake_nonoverlap.f1_score.toFixed(3) : '—';
    document.getElementById('ag-fpr').textContent = pct(ag.false_positive_rate);
    document.getElementById('ag-count').textContent = num(ag.sample_count);
    document.getElementById('bbc-fpr').textContent = pct(bbc.flagged_fake_rate);
    document.getElementById('bbc-count').textContent = num(bbc.sample_count);
    document.getElementById('liar-accuracy').textContent = pct(ext.accuracy);
    document.getElementById('liar-baseline').textContent = pct(ext.majority_baseline_accuracy);
    document.getElementById('liar-balanced').textContent = pct(ext.balanced_accuracy);
    document.getElementById('liar-count').textContent = num(ext.sample_count);
    document.getElementById('dataset-note').textContent = `${num(report.split_sizes?.test)} held-out articles / Trained ${report.training_date || 'date unavailable'}`;
    const curve = report.learning_curve || [];
    if (curve.length) {
      const start = curve[0], end = curve[curve.length - 1];
      const gap = (end.train_article_f1 - end.validation_article_f1) * 100;
      document.getElementById('fit-summary').textContent = `Article validation F1 rose from ${start.validation_article_f1.toFixed(3)} at 25% training data to ${end.validation_article_f1.toFixed(3)} at full data. Final train-to-validation gap: ${gap.toFixed(1)} percentage points.`;
    }
  } catch (cause) {
    const error = document.getElementById('stats-error'); error.textContent = cause.message; error.hidden = false;
  }
}
