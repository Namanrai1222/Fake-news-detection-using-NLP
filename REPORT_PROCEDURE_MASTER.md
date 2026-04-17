# MASTER PROCEDURE FILE FOR AI REPORT GENERATION

Project: Fake News Detection Using NLP and Machine Learning
Version: 1.0
Date: 2026-04-18

This is a single, complete file to generate your full mini project report exactly in the required academic format.
Use this file as input to any AI writer and ask it to create the final report document.

------------------------------------------------------------
## 1) HOW TO USE THIS FILE

1. Fill the placeholders in Section 2 and Section 3.
2. Copy Section 6 (AI MASTER PROMPT) and paste it into your AI tool.
3. Ask AI to generate the full report in one output.
4. Export to DOCX/PDF and do final name/signature/date checks.

Important:
- Do not skip Declaration, Certificate, ToC, List of Tables, List of Figures, List of Abbreviations, References, Appendices, Publications, Plagiarism Report, CV pages.
- Keep all formatting rules exactly.

------------------------------------------------------------
## 2) STUDENT AND INSTITUTE DETAILS (FILL THESE)

Institute Name: NIET Greater Noida
University Name: Dr. A.P.J. Abdul Kalam Technical University, Lucknow
Department: CSE
Degree: B.Tech
Session/Year: [FILL]

Project Title: Fake News Detection Using NLP and Machine Learning with Reliability and Explainability

Student 1 Name: [FILL]
Student 1 Roll No: [FILL]
Student 2 Name: [FILL]
Student 2 Roll No: [FILL]
Student 3 Name: [FILL]
Student 3 Roll No: [FILL]
Student 4 Name: [FILL]
Student 4 Roll No: [FILL]

Supervisor Name: [FILL]
Supervisor Designation: [FILL]
HOD Name: [FILL]
HOD Designation: [FILL]
Submission Date: [FILL]
Place: Greater Noida

------------------------------------------------------------
## 3) PROJECT DATA PACK (ALREADY PRE-FILLED FROM CODEBASE)

### 3.1 Problem Statement
Build a machine learning system that classifies a news article as Fake or Real from textual content, with explainability, confidence, and deployment-ready API support.

### 3.2 Dataset and Labels
- Source files:
  - data/raw/Fake.csv
  - data/raw/True.csv
- Label mapping:
  - Fake = 0
  - Real = 1

### 3.3 Data Cleaning and Filtering Logic
- Remove duplicate rows using text field.
- Remove null text rows.
- Keep rows with 20 to 2000 words.
- Shuffle with random_state=42.
- Advanced text preprocessing:
  - lowercasing
  - punctuation removal
  - stopword filtering
  - tokenization
  - lemmatization (NLTK async load with fallback)
  - removal of source marker patterns like Reuters/AP tags.

### 3.4 Train/Test and Modeling
- Train-test split: 80/20
- Stratified split: Yes
- Vectorizer: TF-IDF (max_features=5000, ngram_range=(1,2))
- Model: Logistic Regression
- Hyperparameters:
  - C = 1.0
  - class_weight = balanced
  - max_iter = 1000
  - random_state = 42
- Cross-validation: 5-fold accuracy check

### 3.5 Final Metrics (from metrics_v2.json)
- Accuracy: 0.9547
- Precision: 0.9922
- Recall: 0.9256
- F1 Score: 0.9577
- Brier Score: 0.0433
- ECE: 0.1065
- Confusion Matrix:
  - [3365, 31]
  - [315, 3921]

### 3.6 Deployment and System Components
- Backend: Flask
- API routes: /predict, /stats, /health
- Optional explanation layer: local Ollama model (non-blocking fallback)
- Explainability: feature signal extraction, optional LIME support
- Security hardening:
  - restricted CORS origins
  - optional API key gate (X-API-Key)
  - request size limit
  - rate limiting
  - minimal health output in production mode

### 3.7 Software Stack
- Python 3.x
- Flask 3.1.0
- flask-cors 5.0.1
- pandas 3.0.1
- numpy 2.4.2
- scikit-learn 1.8.0
- scipy 1.17.1
- nltk 3.9.1
- lime 0.2.0.1
- joblib 1.5.3

------------------------------------------------------------
## 4) COMPULSORY REPORT FORMAT RULES (MATCH TEMPLATE)

Use these exact style constraints while generating report:

- Page size: A4
- Font family: Times New Roman
- Body font size: 12
- Section heading font size: 14 (bold)
- Chapter title font size: 20 (bold, uppercase)
- Line spacing: Single
- Alignment:
  - Body paragraphs: Justified
  - Main headings: Center aligned
- Margin: Standard academic margins (recommended 1 inch on all sides)
- Border: Single rectangular black border around printable area on every page
- Page numbering:
  - Preliminary pages (Declaration to Abbreviations): lower roman (i, ii, iii...)
  - Main chapters onward: arabic (1, 2, 3...)

------------------------------------------------------------
## 5) REQUIRED REPORT SEQUENCE (DO NOT CHANGE ORDER)

1. Declaration
2. Certificate
3. Acknowledgements
4. Abstract
5. Table of Contents
6. List of Tables
7. List of Figures
8. List of Abbreviations
9. Chapter 1: Introduction
10. Chapter 2: Literature Review
11. Chapter 3: Requirements and Analysis
12. Chapter 4: Proposed Methodology
13. Chapter 5: Results
14. Chapter 6: Conclusion and Future Work
15. References (IEEE style)
16. Appendices
17. Publications
18. Plagiarism Report
19. Curriculum Vitae

------------------------------------------------------------
## 6) AI MASTER PROMPT (COPY FROM HERE)

You are an academic report writer. Generate a complete B.Tech mini project report in formal academic language for the project below.

STRICT OUTPUT REQUIREMENTS:
1. Follow the exact section order provided.
2. Use Times New Roman style conventions and spacing guidance in content notes.
3. Do not leave placeholders like [FILL], <text>, or TODO.
4. Use realistic academic wording and maintain consistency across all sections.
5. Add properly numbered headings and subheadings.
6. Include tables and figure captions where relevant.
7. Include citations and references in IEEE format.
8. Keep all signatures/date areas in Declaration and Certificate pages.
9. Include List of Tables, List of Figures, and List of Abbreviations with relevant entries.
10. Ensure chapter content is complete and not generic.

PROJECT AND INSTITUTE DETAILS:
- Institute: NIET Greater Noida
- University: Dr. A.P.J. Abdul Kalam Technical University, Lucknow
- Department: CSE
- Degree: B.Tech
- Project Title: Fake News Detection Using NLP and Machine Learning with Reliability and Explainability
- Session/Year: [USE PROVIDED VALUE OR INFER CURRENT SESSION]
- Student names and roll numbers:
  - Student 1: [FILL BEFORE FINAL EXPORT]
  - Student 2: [FILL BEFORE FINAL EXPORT]
  - Student 3: [FILL BEFORE FINAL EXPORT]
  - Student 4: [FILL BEFORE FINAL EXPORT]
- Supervisor: [FILL BEFORE FINAL EXPORT]
- HOD: [FILL BEFORE FINAL EXPORT]

TECHNICAL FACTS TO USE:
- Dataset files: Fake.csv and True.csv
- Binary labels: Fake=0, Real=1
- Data cleaning: duplicates removal, null removal, word-count filtering (20 to 2000), shuffle
- Preprocessing: lowercase, punctuation removal, tokenization, stopword removal, lemmatization
- Train-test split: 80/20 stratified
- Feature extraction: TF-IDF (max_features=5000, ngram_range=(1,2))
- Classifier: Logistic Regression (C=1.0, class_weight=balanced, max_iter=1000)
- Validation: 5-fold cross-validation
- Metrics:
  - Accuracy 95.47%
  - Precision 99.22%
  - Recall 92.56%
  - F1 Score 95.77%
  - Brier Score 0.0433
  - ECE 0.1065
  - Confusion Matrix: [[3365,31],[315,3921]]
- Backend: Flask API with routes /predict, /stats, /health
- Explainability and reliability enhancements:
  - confidence calibration support
  - feature-level signal reasoning
  - optional local LLM explanation layer via Ollama
- Security controls:
  - restricted CORS
  - optional API key access
  - request-size limit
  - rate limiting
  - production-safe health response

CONTENT EXPECTATIONS PER CHAPTER:
- Chapter 1: background, motivation, objectives, scope, report organization
- Chapter 2: literature review of fake news detection, NLP methods, ML and DL approaches, identified research gaps
- Chapter 3: requirement specification, software/hardware requirements, planning and scheduling, preliminary product description
- Chapter 4: full methodology from data collection to preprocessing, vectorization, model training, evaluation, API integration, explainability and security architecture
- Chapter 5: experimental results, metric interpretation, confusion matrix discussion, strengths/limitations, comparative discussion
- Chapter 6: conclusion, contributions, practical impact, future improvements

EXTRA PAGES:
- Declaration page with 4 student signature blocks
- Certificate page with Supervisor and HOD signature blocks
- Appendices page describing what is attached (code snippets, logs, screenshots, extra results)
- Publications page (if none, mention that publication is under review or not applicable)
- Plagiarism report page with placeholder sentence for final percentage insertion
- One-page Curriculum Vitae format for student (repeat format for all if required)

REFERENCING:
- Use IEEE style references.
- Add at least 10 relevant references for fake news detection, NLP, TF-IDF, logistic regression, explainable AI, and reliability calibration.

Now generate the full report content in final polished form.

## END OF AI MASTER PROMPT

------------------------------------------------------------
## 7) READY-MADE ABBREVIATIONS (USE IN REPORT)

- NLP: Natural Language Processing
- TF-IDF: Term Frequency-Inverse Document Frequency
- ML: Machine Learning
- API: Application Programming Interface
- LLM: Large Language Model
- ECE: Expected Calibration Error
- CV: Cross Validation
- LR: Logistic Regression
- UI: User Interface
- JSON: JavaScript Object Notation

------------------------------------------------------------
## 8) READY-MADE TABLES/LISTS SEEDS

Suggested List of Tables entries:
- Dataset composition and label distribution
- Data cleaning and filtering summary
- Model hyperparameter configuration
- Performance metrics summary
- Confusion matrix values
- Security control checklist

Suggested List of Figures entries:
- System architecture diagram
- Data preprocessing pipeline flowchart
- Model training pipeline
- API request-response flow
- Confusion matrix heatmap
- Reliability/calibration trend chart

------------------------------------------------------------
## 9) FINAL QUALITY CHECKLIST BEFORE SUBMISSION

- Student names and roll numbers are correct on all required pages.
- Supervisor and HOD names/designations are correct.
- Signature blocks exist in Declaration and Certificate pages.
- All chapters are complete and consistent with project implementation.
- Metrics in text match exact values from Section 3.5.
- References follow IEEE format.
- Page numbering and order are correct.
- Borders and typography are consistent with format.
- Plagiarism page includes final percentage from official report.
- CV page is updated with latest student details.

------------------------------------------------------------
END OF FILE
