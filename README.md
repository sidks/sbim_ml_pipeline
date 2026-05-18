# SBIM ML Pipeline

This repository contains the machine learning pipeline used for training personalized emotion prediction models and generating participant-level PDF reports from passive sensing and behavioral data collected through the CINGO platform.

The pipeline:

- Loads and preprocesses behavioral feature data
- Trains participant-specific emotion prediction models
- Computes SHAP explainability values
- Generates visualizations and personalized PDF reports

---

# Repository Structure

```text
sbim_ml_pipeline/
│
├── training_report.py
├── requirements.txt
├── report_template.pdf
├── integerfeatures.csv
├── generalfeatures.csv
├── accelerationfeatures.csv
│
├── overlay_reports/
│   ├── <participant>_final.pdf
│   └── ...
│
│
└── README.md
