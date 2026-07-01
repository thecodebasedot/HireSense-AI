#!/usr/bin/env bash
# HireSense AI — run the whole pipeline with one command.
# End-to-end: install deps -> generate data -> train -> demo prediction
set -e

cd "$(dirname "$0")"

echo "==> Installing dependencies"
pip install -q -r requirements.txt

echo "==> Generating synthetic applicant dataset"
python src/generate_data.py

echo "==> Training SVM model"
python src/train.py

echo "==> Generating evaluation plots"
python src/visualize.py

echo "==> Comparing SVM against other models"
python src/compare_models.py

echo "==> Training deep learning (neural network) model"
python src/deep_model.py

echo "==> Fairness / bias audit"
python src/fairness.py

echo "==> Bias mitigation (group-specific thresholds)"
python src/bias_mitigation.py

echo "==> Optimal decision threshold"
python src/threshold.py

echo "==> Calibration diagnostics"
python src/diagnostics.py

echo "==> Conformal prediction (uncertainty)"
python src/conformal.py

echo "==> Data drift check (demo)"
python src/drift.py --demo

echo "==> Generating model card"
python src/model_card.py

echo "==> Demo: screening a strong applicant"
python src/predict.py \
    --years-experience 6 --education-level 2 \
    --skill-match-score 85 --interview-score 80 \
    --communication-score 78 --num-certifications 3 \
    --num-projects 8 --gpa 3.7

echo "==> Demo: screening a weak applicant"
python src/predict.py \
    --years-experience 0 --education-level 0 \
    --skill-match-score 30 --interview-score 35 \
    --communication-score 40 --num-certifications 0 \
    --num-projects 0 --gpa 2.4
