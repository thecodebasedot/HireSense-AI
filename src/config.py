"""
HireSense AI — central configuration.

Feature schema, paths, and constants shared across the data generator,
trainer, and predictor.
"""
from pathlib import Path

# --- Project paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

DATASET_PATH = DATA_DIR / "applicants.csv"
MODEL_PATH = MODEL_DIR / "hiresense_svm.joblib"

# --- Feature schema ---
# Numeric features used to describe a job applicant.
# Order matters: predictions accept inputs in this order.
NUMERIC_FEATURES = [
    "years_experience",     # relevant work experience (years)
    "education_level",      # 0=High School, 1=Bachelor, 2=Master, 3=PhD
    "skill_match_score",    # how well skills match the job requirements (0-100)
    "interview_score",      # interview performance (0-100)
    "communication_score",  # communication skills (0-100)
    "num_certifications",   # number of relevant certifications
    "num_projects",         # number of completed projects
    "gpa",                  # grade point average (0-4)
]

TARGET = "shortlisted"       # 1 = shortlisted, 0 = rejected

EDUCATION_LABELS = {
    0: "High School",
    1: "Bachelor",
    2: "Master",
    3: "PhD",
}

RANDOM_STATE = 42
