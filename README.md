# HireSense AI 🧠💼

**A job applicant screening system built with a Support Vector Machine (SVM).**

HireSense AI is a machine-learning system that automatically screens job
applicants. It analyses a candidate's experience, education, skills, interview
scores and more, then predicts whether the candidate should be **shortlisted**
or **rejected**.

---

## 🎯 Overview

| Item | Details |
|------|---------|
| **Project Name** | HireSense AI |
| **Problem Solved** | Job Applicant Screening |
| **ML Algorithm** | Support Vector Machine (SVM) |
| **Type** | Binary Classification (Shortlist / Reject) |

**Why SVM?** Applicant features are numeric and the boundary between classes
can be non-linear. Through the kernel trick, an SVM learns such boundaries well
and is efficient on small-to-medium datasets.

---

## 📊 Features

The model uses the following eight attributes of each applicant:

| Feature | Meaning | Range |
|---------|---------|-------|
| `years_experience` | Relevant work experience (years) | 0–25 |
| `education_level` | Education level (0=HS, 1=Bachelor, 2=Master, 3=PhD) | 0–3 |
| `skill_match_score` | How well skills match the job requirements | 0–100 |
| `interview_score` | Interview performance | 0–100 |
| `communication_score` | Communication skills | 0–100 |
| `num_certifications` | Number of relevant certifications | 0–10 |
| `num_projects` | Number of completed projects | 0–20 |
| `gpa` | Grade point average | 2.0–4.0 |

**Target:** `shortlisted` → `1` (shortlist) or `0` (reject)

---

## 🗂️ Project Structure

```
HireSense-AI/
├── src/
│   ├── config.py         # feature schema, paths, constants
│   ├── generate_data.py  # synthetic applicant dataset generator
│   ├── train.py          # SVM training + hyperparameter tuning
│   └── predict.py        # screen new applicants (single / batch)
├── data/
│   └── sample_applicants.csv   # example input for batch scoring
├── models/               # trained model is saved here
├── tests/
│   └── test_pipeline.py  # smoke tests
├── run.sh                # full pipeline in one command
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Everything in one command

```bash
bash run.sh
```

This installs dependencies → generates data → trains the model → shows demo
predictions.

### 2. Step by step

```bash
# Install dependencies
pip install -r requirements.txt

# 1) Generate the synthetic dataset
python src/generate_data.py            # default: 2000 applicants
python src/generate_data.py -n 5000    # custom count

# 2) Train the SVM model
python src/train.py

# 3) Screen a single applicant
python src/predict.py \
    --years-experience 6 --education-level 2 \
    --skill-match-score 85 --interview-score 80 \
    --communication-score 78 --num-certifications 3 \
    --num-projects 8 --gpa 3.7

# 4) Screen many applicants from a CSV
python src/predict.py --csv data/sample_applicants.csv
```

---

## 🧪 Running Tests

```bash
python tests/test_pipeline.py
# or, if pytest is installed:
python -m pytest tests/ -q
```

---

## ⚙️ How It Works

1. **Preprocessing** — all features are standardised with `StandardScaler`
   (SVMs are distance-based, so scaling is essential).
2. **Model** — `SVC` (RBF and linear kernels), tuned with `GridSearchCV` over
   `C`, `gamma` and the kernel using 5-fold cross-validation (`roc_auc`
   scoring).
3. **Calibration** — `CalibratedClassifierCV` produces reliable probabilities
   so each decision comes with a confidence score.
4. **Output** — for each applicant, a decision (`Shortlist` / `Reject`) and the
   probability of being shortlisted.

### 📈 Sample Metrics

On a synthetic 2000-sample dataset (held-out test set):

```
Accuracy : ~0.82
ROC-AUC  : ~0.90
```

> Note: no real hiring data is used here (it is sensitive and private).
> Instead, a realistic synthetic dataset is generated so the whole pipeline
> runs reproducibly. To use real data, provide it as `data/applicants.csv`
> with the same columns and run `python src/train.py` directly.

---

## 📄 License

This project is released under the repository's [LICENSE](LICENSE) file.
