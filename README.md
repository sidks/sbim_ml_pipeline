# SBIM ML Pipeline

This repository contains the machine learning pipeline used for training personalized emotion prediction models and generating participant-level PDF reports from behavioral data collected through the CINGO platform.

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
├── plots/
│
└── README.md
```

---

# Environment Setup

## 1. Clone the repository

```bash
cd /srv/repos

git clone git@github.com:sidks/sbim_ml_pipeline.git

cd sbim_ml_pipeline
```

---

## 2. Create and activate a virtual environment

```bash
python3 -m venv /opt/envs/sbim-ml

source /opt/envs/sbim-ml/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Input Data

The script expects a CSV dataset generated from the CINGO pipeline.

Example:

```python
df = pd.read_csv(
    "/srv/repos/raddlab_datascience/cingo-pipeline/output/<filename>.csv"
)
```

Update the filename/path inside `training_report.py` before execution.

---

# Supporting Files

The following files must exist in the repository root:

| File | Purpose |
|---|---|
| `integerfeatures.csv` | Integer feature translations and metadata |
| `generalfeatures.csv` | General feature translations and metadata |
| `accelerationfeatures.csv` | Acceleration feature translations and metadata |
| `report_template.pdf` | Base PDF template used for report generation |

---

# Running the Pipeline

Activate the environment:

```bash
source /opt/envs/sbim-ml/bin/activate
```

Run the script:

```bash
python training_report.py
```

---

# Notes

- The script is computationally intensive and may take significant time for large datasets.
- GPU acceleration is not currently required.
- Reports are generated using ReportLab and merged with a PDF template using PyPDF2.

---

# Example Workflow

## Generate data from the CINGO pipeline

```bash
cd /srv/repos/raddlab_datascience/cingo-pipeline

python extract-data-for-ai-models.py \
    --days-back 7 \
    --study-name "Mock Study 1"
```

## Run the ML pipeline

```bash
cd /srv/repos/sbim_ml_pipeline

source /opt/envs/sbim-ml/bin/activate

python training_report.py
```

---
