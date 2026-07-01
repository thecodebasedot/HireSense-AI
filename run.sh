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
