# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

People reviewing English news articles who want a quick language-based lead before checking sources. This is inferred from the existing app and task, not a validated user study.

## Product Purpose

Classify an article with a reproducible text model, reveal the terms that influenced its estimate, and show the evidence and limits of evaluation.

## Capabilities and Constraints

The Flask service accepts 20–2,500 words. Google Fact Check and local Ollama are optional. The classifier is trained on historical ISOT and nonoverlapping WELFake articles, plus weak ordinary-news examples from AG News. It has weak transfer to LIAR political claims and cannot establish factual truth by itself.

## Evidence on Hand

Bundled ISOT and LIAR datasets; pinned WELFake, AG News, and BBC downloads; held-out evaluation in `models/metrics_v2.json`. AG News and BBC have topic labels, not independent truth verdicts. No validated live-news benchmark or real user research is available.

## Product Principles

- Show measured performance with its test domain.
- State uncertainty in plain language.
- Keep the article workflow fast without optional services.
