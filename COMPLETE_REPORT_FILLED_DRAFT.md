# FAKE NEWS DETECTION USING NLP AND MACHINE LEARNING WITH RELIABILITY AND EXPLAINABILITY

## DECLARATION

We hereby declare that the work presented in this report entitled "Fake News Detection Using NLP and Machine Learning with Reliability and Explainability" was carried out by us. This report has not been submitted in part or full to any other university or institute for the award of any degree or diploma. We have duly acknowledged all the reference materials, published literature, websites, and tools used in this work. The implementation, experimentation, and analysis described in this report are based on our original effort under the guidance of the project supervisor.

We further declare that no part of this report is plagiarized and no data, result, or graph has been fabricated or manipulated. We accept full responsibility for the authenticity of the content and results presented in this project.

Name: ____________________    Roll Number: ____________________
Candidate Signature: ____________________

Name: ____________________    Roll Number: ____________________
Candidate Signature: ____________________

Name: ____________________    Roll Number: ____________________
Candidate Signature: ____________________

Name: ____________________    Roll Number: ____________________
Candidate Signature: ____________________

---

## CERTIFICATE

This is to certify that the mini project report entitled "Fake News Detection Using NLP and Machine Learning with Reliability and Explainability" submitted by the students listed below, in partial fulfillment of the requirements for the award of Bachelor of Technology in Computer Science and Engineering from Dr. A.P.J. Abdul Kalam Technical University, Lucknow, is a bonafide record of work carried out by them under our supervision.

The project report embodies the results of original work and has been completed to our satisfaction.

Student 1: ____________________ (Roll No: ____________________)
Student 2: ____________________ (Roll No: ____________________)
Student 3: ____________________ (Roll No: ____________________)
Student 4: ____________________ (Roll No: ____________________)

Supervisor Signature: ____________________
Name of Supervisor: ____________________
Designation: ____________________
Department: CSE
Institute: NIET Greater Noida

HOD Signature: ____________________
Name of HOD: ____________________
Designation: ____________________
Department: CSE
Institute: NIET Greater Noida

Date: ____________________

---

## ACKNOWLEDGEMENTS

We express our sincere gratitude to our supervisor for continuous guidance, motivation, and constructive suggestions throughout the project. Their technical inputs helped us in designing a robust fake news detection system and in improving the overall quality of this work.

We are thankful to the Head of Department, Computer Science and Engineering, NIET Greater Noida, for providing the academic environment and facilities required for this project.

We also acknowledge our faculty members, classmates, and family members for their encouragement and support during the project implementation and documentation phases.

---

## ABSTRACT

The rapid spread of misinformation through online platforms has created serious social, political, and economic challenges. This project presents a practical machine learning system for automatic fake news detection using Natural Language Processing (NLP). The proposed system classifies a given news article as Fake or Real using a TF-IDF feature representation and a Logistic Regression classifier.

The dataset is prepared by combining Fake and True news sources, followed by duplicate removal, null handling, and quality filtering based on article length. Text preprocessing includes lowercasing, punctuation removal, stopword elimination, tokenization, and lemmatization. The model is trained using stratified train-test splitting and evaluated using standard metrics.

To improve trust and usability, the system includes reliability and explainability features such as calibrated confidence handling, feature-level evidence signals, and an optional local LLM-based explanation layer using Ollama. The backend is implemented using Flask APIs and is secured with CORS allowlisting, optional API key authentication, rate limiting, and request size restrictions.

The final model achieved strong performance: Accuracy 95.47%, Precision 99.22%, Recall 92.56%, and F1-score 95.77%. These results show that the system is suitable for real-world deployment scenarios requiring fast, interpretable, and secure fake news detection.

Keywords: Fake News Detection, NLP, TF-IDF, Logistic Regression, Explainable AI, Calibration, Flask API.

---

## TABLE OF CONTENTS

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
15. References
16. Appendices
17. Publications
18. Plagiarism Report
19. Curriculum Vitae

---

## LIST OF TABLES

Table 1: Dataset and label mapping
Table 2: Data cleaning and filtering summary
Table 3: Model configuration and hyperparameters
Table 4: Final performance metrics
Table 5: Confusion matrix
Table 6: Security controls in API layer

---

## LIST OF FIGURES

Figure 1: Overall system architecture
Figure 2: Data preprocessing pipeline
Figure 3: Model training and evaluation flow
Figure 4: API request-response flow
Figure 5: Confusion matrix visualization
Figure 6: Reliability and confidence interpretation flow

---

## LIST OF ABBREVIATIONS

NLP - Natural Language Processing
ML - Machine Learning
DL - Deep Learning
TF-IDF - Term Frequency-Inverse Document Frequency
LR - Logistic Regression
API - Application Programming Interface
LLM - Large Language Model
ECE - Expected Calibration Error
CV - Cross Validation
JSON - JavaScript Object Notation

---

## CHAPTER 1: INTRODUCTION

### 1.1 Background
In recent years, digital media has become the primary source of news for most users. Social media platforms and online portals allow information to spread very quickly. While this improves accessibility, it also enables rapid propagation of fake and misleading content. Such misinformation can influence public opinion, create panic, and affect democratic processes.

### 1.2 Problem Statement
Manual verification of large volumes of online news is not feasible in real time. Therefore, an automated approach is required to classify news content based on textual patterns and linguistic features.

### 1.3 Objectives
1. To build a binary classification model for fake news detection.
2. To design a preprocessing pipeline for robust text normalization.
3. To evaluate model performance using standard classification metrics.
4. To provide explainable outputs for better user trust.
5. To deploy the model through secure and scalable Flask APIs.

### 1.4 Scope
The project focuses on English textual news articles and performs binary classification (Fake or Real). It does not perform image/video verification or multilingual analysis in the current version.

### 1.5 Project Report Organization
- Chapter 1 introduces the problem and goals.
- Chapter 2 discusses existing literature.
- Chapter 3 defines requirements and analysis.
- Chapter 4 explains the proposed methodology.
- Chapter 5 presents results and discussion.
- Chapter 6 concludes the work and future scope.

---

## CHAPTER 2: LITERATURE REVIEW

### 2.1 Overview
Fake news detection has been studied using both traditional machine learning and deep learning techniques. Most approaches rely on linguistic features, contextual signals, user behavior, or source credibility.

### 2.2 Traditional NLP and ML Methods
Classical techniques such as Naive Bayes, Support Vector Machines, and Logistic Regression with TF-IDF features are commonly used due to interpretability and low computational cost. These methods often perform well on balanced text datasets.

### 2.3 Deep Learning-Based Methods
Recent studies use CNN, RNN, BiLSTM, and transformer-based models (e.g., BERT). These models capture semantic relations better but need larger datasets and higher compute resources.

### 2.4 Explainability and Reliability in Detection Systems
High accuracy alone is not enough for sensitive applications. Explainability methods such as LIME and feature attribution improve transparency. Calibration metrics (Brier score and ECE) help measure confidence reliability.

### 2.5 Research Gaps Identified
1. Many systems report high accuracy but provide limited explanation.
2. Confidence values are often uncalibrated.
3. Production APIs are frequently built without strong security controls.
4. Lightweight deployable systems are less explored compared to heavy transformer models.

### 2.6 Contribution of This Work
This project addresses the above gaps by combining strong baseline ML performance with reliability indicators, optional LLM explanation, and secure API design.

---

## CHAPTER 3: REQUIREMENTS AND ANALYSIS

### 3.1 Requirements Specification
The system should:
1. Accept input news text.
2. Clean and transform text into features.
3. Predict Fake/Real class with confidence.
4. Return interpretable evidence.
5. Expose outputs via REST APIs.

### 3.2 Software Requirements
- Operating System: Windows/Linux
- Language: Python 3.x
- Framework: Flask 3.1.0
- Libraries: pandas, numpy, scikit-learn, scipy, nltk, lime, joblib, flask-cors

### 3.3 Hardware Requirements
- Processor: Intel i5 or equivalent
- RAM: 8 GB minimum (16 GB recommended)
- Storage: 2 GB free for dataset, models, logs, and environment
- Internet: required only for first-time dependency/resource downloads

### 3.4 Dataset and Data Analysis
- Inputs: data/raw/Fake.csv and data/raw/True.csv
- Labels assigned:
  - Fake -> 0
  - Real -> 1
- Cleaning steps:
  - duplicate text removal
  - null removal
  - word-count filtering (20 to 2000 words)
- Data split strategy: stratified 80/20 train-test

### 3.5 Planning and Scheduling
Phase 1: Data loading and cleaning
Phase 2: Text preprocessing and feature extraction
Phase 3: Model training and validation
Phase 4: API integration and frontend connection
Phase 5: Explainability, reliability, and security hardening
Phase 6: Testing and report documentation

### 3.6 Preliminary Product Description
The final product is a web-enabled fake news analysis tool with:
- backend classification engine
- confidence and reliability outputs
- optional LLM-generated explanation
- metrics dashboard support

---

## CHAPTER 4: PROPOSED METHODOLOGY

### 4.1 System Architecture
The architecture includes four layers:
1. Data Layer: CSV dataset ingestion and cleaning.
2. NLP Layer: text normalization and token processing.
3. ML Layer: TF-IDF vectorization and Logistic Regression classification.
4. Service Layer: Flask APIs for prediction and metrics.

### 4.2 Data Loading and Labeling
The fake and true datasets are loaded using pandas, labeled as 0 and 1 respectively, and merged into a single frame for downstream processing.

### 4.3 Text Preprocessing Pipeline
The pipeline performs:
1. removal of source markers (Reuters/AP style tags)
2. conversion to lowercase
3. punctuation removal
4. tokenization
5. stopword filtering
6. lemmatization

An asynchronous NLTK loading approach with fallback preprocessing is used to keep startup responsive.

### 4.4 Feature Engineering
Text is transformed using TF-IDF with:
- max_features = 5000
- ngram_range = (1, 2)

This captures unigram and bigram context while controlling model dimensionality.

### 4.5 Model Training
Classifier used: Logistic Regression
- C = 1.0
- class_weight = balanced
- max_iter = 1000
- random_state = 42

A 5-fold cross-validation check is performed before final fitting.

### 4.6 Evaluation Strategy
The model is evaluated on the held-out test set using:
- accuracy
- precision
- recall
- F1-score
- confusion matrix
- Brier score
- Expected Calibration Error (ECE)

### 4.7 Explainability and Reasoning Layer
The system provides:
- top lexical feature signals influencing class
- confidence and reliability banding
- optional local LLM explanation via Ollama
- non-blocking fallback if LLM service is unavailable

### 4.8 API and Security Methodology
Primary routes:
- /predict
- /stats
- /health

Security controls implemented:
1. restricted CORS allowlist
2. optional X-API-Key validation
3. request payload size limit
4. per-client rate limiting
5. minimal health response in production mode

---

## CHAPTER 5: RESULTS AND DISCUSSION

### 5.1 Final Performance Metrics
Table: Final Model Metrics
- Accuracy: 0.9547
- Precision: 0.9922
- Recall: 0.9256
- F1 Score: 0.9577
- Brier Score: 0.0433
- ECE: 0.1065

### 5.2 Confusion Matrix
Confusion Matrix:
- True Negative: 3365
- False Positive: 31
- False Negative: 315
- True Positive: 3921

### 5.3 Interpretation
The model achieves very high precision, indicating that when it predicts Real news, it is usually correct. Recall is slightly lower than precision, showing some real-news items may still be misclassified. Overall F1 score confirms strong balance.

Low Brier score and acceptable ECE suggest confidence estimates are reasonably reliable.

### 5.4 Strengths
1. High classification performance with lightweight architecture.
2. Fast inference suitable for real-time APIs.
3. Interpretable output with feature-level signals.
4. Improved reliability through calibration-aware metrics.
5. Production-oriented security controls.

### 5.5 Limitations
1. English-language focus only.
2. Text-only analysis (no image/video metadata).
3. Domain shift may reduce performance on unseen news styles.
4. LLM explanation quality depends on local model availability.

### 5.6 Comparative Discussion
Compared to heavy deep learning systems, the proposed solution offers a strong trade-off between speed, interpretability, and deployment simplicity while still maintaining high accuracy.

---

## CHAPTER 6: CONCLUSION AND FUTURE WORK

### 6.1 Conclusion
This project successfully develops a robust fake news detection framework using NLP and machine learning. By combining TF-IDF features with Logistic Regression, the system achieves high predictive performance while keeping computation affordable. The integration of explainability, reliability indicators, and secure API design makes the solution practical for deployment.

### 6.2 Key Contributions
1. End-to-end fake news detection pipeline.
2. Clean and reliable training/evaluation setup.
3. Confidence-aware and explainable predictions.
4. Secure and extensible Flask-based inference service.

### 6.3 Future Work
1. Add multilingual and code-mixed text support.
2. Integrate transformer-based comparative models.
3. Add source credibility graph and temporal propagation signals.
4. Incorporate multimodal verification (image/video).
5. Build continuous learning pipeline with feedback loops.

---

## REFERENCES (IEEE STYLE)

[1] H. Allcott and M. Gentzkow, "Social Media and Fake News in the 2016 Election," Journal of Economic Perspectives, vol. 31, no. 2, pp. 211-236, 2017.

[2] N. J. Conroy, V. L. Rubin, and Y. Chen, "Automatic Deception Detection: Methods for Finding Fake News," Proceedings of the Association for Information Science and Technology, vol. 52, no. 1, pp. 1-4, 2015.

[3] V. L. Rubin, N. J. Conroy, and Y. Chen, "Towards News Verification: Deception Detection Methods for News Discourse," in Proc. Hawaii International Conference on System Sciences, 2015.

[4] P. Bojanowski, E. Grave, A. Joulin, and T. Mikolov, "Enriching Word Vectors with Subword Information," Transactions of the ACL, vol. 5, pp. 135-146, 2017.

[5] T. Mikolov, K. Chen, G. Corrado, and J. Dean, "Efficient Estimation of Word Representations in Vector Space," arXiv preprint arXiv:1301.3781, 2013.

[6] J. Devlin, M. W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in Proc. NAACL-HLT, 2019.

[7] M. T. Ribeiro, S. Singh, and C. Guestrin, "Why Should I Trust You?: Explaining the Predictions of Any Classifier," in Proc. ACM SIGKDD, pp. 1135-1144, 2016.

[8] T. Fawcett, "An Introduction to ROC Analysis," Pattern Recognition Letters, vol. 27, no. 8, pp. 861-874, 2006.

[9] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in Proc. ICML, pp. 1321-1330, 2017.

[10] D. Jurafsky and J. H. Martin, Speech and Language Processing, 3rd ed. draft, 2023.

[11] S. Raschka and V. Mirjalili, Python Machine Learning, 3rd ed., Packt, 2019.

[12] A. Géron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd ed., O'Reilly, 2022.

---

## APPENDICES

### Appendix A: Project Folder Structure
- app/
- src/
- data/raw/
- data/processed/
- models/
- frontend/
- tests/
- logs/

### Appendix B: API Contracts
- POST /predict: accepts text, optional flags; returns class, confidence, explainability outputs
- GET /stats: returns metrics summary
- GET /health: service and model readiness

### Appendix C: Security Configuration Variables
- APP_API_KEY
- ALLOWED_ORIGINS
- MAX_REQUEST_BYTES
- RATE_LIMIT_WINDOW_SECONDS
- RATE_LIMIT_MAX_REQUESTS
- HEALTH_DETAILS

### Appendix D: Additional Artifacts
- model and vectorizer files
- calibration artifact
- metrics JSON
- log samples

---

## PUBLICATIONS

Current status: No external publication has been formally accepted at the time of this report submission.

Proposed statement for final copy:
"A paper based on this work is under preparation/submission to a suitable conference or journal."

---

## PLAGIARISM REPORT

This section should include the institute-approved plagiarism summary.

Final declaration format:
"The plagiarism similarity index for this report is ______ %, which is within the permissible academic limit prescribed by the institute."

(Attach official plagiarism report as annexure in final bound copy.)

---

## CURRICULUM VITAE (ONE PAGE TEMPLATE)

### Student Name
Address: ____________________
Phone: ____________________
Email: ____________________

Career Objective:
To apply knowledge of machine learning, NLP, and software engineering to build reliable and secure real-world AI systems.

Academic Qualifications:
- B.Tech (CSE), NIET Greater Noida, AKTU - [Year], [CGPA]
- Class XII - [Board], [Year], [Percentage]
- Class X - [Board], [Year], [Percentage]

Technical Skills:
- Languages: Python, SQL, Java (basic)
- Libraries: scikit-learn, pandas, numpy, nltk, flask
- Tools: Git, VS Code, Postman

Projects:
- Fake News Detection Using NLP and Machine Learning with Reliability and Explainability

Internships/Training:
- [If applicable]

Achievements:
- [If applicable]

Declaration:
I hereby declare that the above information is true to the best of my knowledge.

Date: ____________________
Place: ____________________
Signature: ____________________

(Repeat CV page for each student if required by department format.)

---

## END OF COMPLETE REPORT DRAFT
