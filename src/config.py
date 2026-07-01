"""
HireSense AI — কেন্দ্রীয় কনফিগারেশন
Central configuration: feature schema, paths, and constants shared across
the data generator, trainer, and predictor.
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
    "years_experience",     # প্রাসঙ্গিক কাজের অভিজ্ঞতা (বছর)
    "education_level",       # শিক্ষাগত স্তর: 0=HS, 1=Bachelor, 2=Master, 3=PhD
    "skill_match_score",     # চাকরির প্রয়োজনের সাথে দক্ষতার মিল (0-100)
    "interview_score",       # ইন্টারভিউ স্কোর (0-100)
    "communication_score",   # যোগাযোগ দক্ষতা (0-100)
    "num_certifications",    # প্রাসঙ্গিক সার্টিফিকেশন সংখ্যা
    "num_projects",          # সম্পন্ন প্রজেক্ট সংখ্যা
    "gpa",                   # সিজিপিএ (0-4)
]

TARGET = "shortlisted"       # 1 = শর্টলিস্টেড, 0 = বাতিল

EDUCATION_LABELS = {
    0: "High School",
    1: "Bachelor",
    2: "Master",
    3: "PhD",
}

RANDOM_STATE = 42
