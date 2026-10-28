import json
import subprocess
import sys
from unittest.mock import patch

from app import app as backend


def test_predict_rejects_non_string_text():
    with patch.object(backend, 'model_loading', False), patch.object(backend, 'predict_fn', lambda text: ('Real', 0.8)):
        response = backend.app.test_client().post('/predict', json={'text': 42})
    assert response.status_code == 400
    assert response.json['error'] == "'text' must be a string."


def test_predict_uses_calibrated_model_without_heuristic_blend():
    article = ' '.join(['A verified report describes an event with evidence'] * 5)
    with patch.object(backend, 'model_loading', False), patch.object(backend, 'predict_fn', lambda text: ('Real', 0.8)), patch.object(backend, '_predictor_obj', None):
        response = backend.app.test_client().post('/predict', json={'text': article, 'use_llm': False, 'use_ensemble': True})
    assert response.status_code == 200
    assert response.json['prediction'] == 'Real'
    assert response.json['confidence'] == 0.8
    assert 'ensemble' not in response.json


def test_stats_uses_saved_counts_and_date():
    with open('models/metrics_v2.json') as file:
        expected = json.load(file)
    response = backend.app.test_client().get('/stats')
    assert response.status_code == 200
    assert response.json['total_articles'] == expected['total_articles']
    assert response.json['training_date'] == expected['training_date']


def test_predict_rejects_non_object_json():
    with patch.object(backend, 'model_loading', False), patch.object(backend, 'predict_fn', lambda text: ('Real', 0.8)):
        response = backend.app.test_client().post('/predict', json=['text'])
    assert response.status_code == 400


def test_security_headers_present():
    response = backend.app.test_client().get('/')
    assert response.headers['X-Content-Type-Options'] == 'nosniff'
    assert "frame-ancestors 'none'" in response.headers['Content-Security-Policy']


def test_ollama_rejects_remote_url():
    with patch.object(backend, 'OLLAMA_URL', 'https://example.com/api/generate'):
        assert backend._get_ollama_explanation('sample', 'Real', 0.8, 'Low', True) is None


def test_wsgi_import_does_not_write_to_read_only_deployment():
    probe = '''
import os
from unittest.mock import patch

exists = os.path.exists
makedirs = os.makedirs

def read_only_exists(path):
    return False if str(path).endswith(os.sep + 'logs') else exists(path)

def read_only_makedirs(path, *args, **kwargs):
    if str(path).endswith(os.sep + 'logs'):
        raise PermissionError('deployment filesystem is read-only')
    return makedirs(path, *args, **kwargs)

with patch('os.path.exists', side_effect=read_only_exists), patch('os.makedirs', side_effect=read_only_makedirs):
    from app.app import app
assert app is not None
'''
    result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
