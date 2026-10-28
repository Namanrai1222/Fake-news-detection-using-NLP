"""Download the pinned research datasets used for retraining and stress tests."""

import hashlib
import os
from pathlib import Path

import requests

from src.config import BASE_DIR

SOURCES = {
    'welfake.parquet': (
        'https://huggingface.co/datasets/davanstrien/WELFake/resolve/386d094f43dde8ba2748cdae4f4e9d8269e88018/data/train-00000-of-00001-290868f0a36350c5.parquet',
        '17f14afc762138b1a724d3ec37a3fc2647ea49e3878e06741073fd82c91dac90',
    ),
    'ag_news_train.parquet': (
        'https://huggingface.co/datasets/fancyzhx/ag_news/resolve/eb185aade064a813bc0b7f42de02595523103ca4/data/train-00000-of-00001.parquet',
        'fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee',
    ),
    'ag_news_test.parquet': (
        'https://huggingface.co/datasets/fancyzhx/ag_news/resolve/eb185aade064a813bc0b7f42de02595523103ca4/data/test-00000-of-00001.parquet',
        '71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb',
    ),
    'bbc-fulltext.zip': (
        'http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip',
        'a351031a62325b3d7ff13146eeb035c22feca8290913b5183d1ff3be35109495',
    ),
}


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    directory = Path(BASE_DIR) / 'data' / 'external'
    directory.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected) in SOURCES.items():
        destination = directory / filename
        if destination.exists() and digest(destination) == expected:
            print(f'{filename}: verified cache')
            continue
        temporary = destination.with_suffix(destination.suffix + '.part')
        temporary.unlink(missing_ok=True)
        print(f'Downloading {filename}...', flush=True)
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with temporary.open('wb') as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    output.write(chunk)
        if digest(temporary) != expected:
            temporary.unlink(missing_ok=True)
            raise ValueError(f'{filename}: SHA-256 mismatch')
        os.replace(temporary, destination)
        print(f'{filename}: verified')


if __name__ == '__main__':
    main()
