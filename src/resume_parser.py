"""
HireSense AI — heuristic resume parser.

Extracts the model's features from a plain-text resume using simple rules and
keyword matching (no heavy NLP dependency). It is intentionally transparent:
every extracted value can be traced to a rule below.

Some features (interview_score, communication_score) cannot be read from a
resume — they come from later hiring stages — so sensible defaults are used
and can be overridden.

Usage:
    python src/resume_parser.py data/sample_resume.txt \
        --job-skills python,sql,machine-learning
"""
import argparse
import re

import pandas as pd

from config import NUMERIC_FEATURES

EDUCATION_KEYWORDS = {
    3: ["ph.d", "phd", "doctorate", "doctoral"],
    2: ["master", "m.sc", "msc", "m.s.", "mba", "m.tech"],
    1: ["bachelor", "b.sc", "bsc", "b.s.", "b.tech", "be "],
    0: ["high school", "secondary", "hsc", "diploma"],
}

CERT_KEYWORDS = ["certified", "certificate", "certification", "aws certified",
                 "pmp", "scrum", "azure", "google cloud", "coursera", "udemy"]


def extract_years_experience(text: str) -> float:
    """Look for 'X years of experience'; else infer from year ranges."""
    m = re.search(r"(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?experience", text)
    if m:
        return float(m.group(1))
    # Fall back to summing explicit year ranges like "2018 - 2022".
    ranges = re.findall(r"(19|20)\d{2}\s*[-–to]+\s*((?:19|20)\d{2}|present|now)", text)
    total = 0
    for start, end in ranges:
        start_year = int(re.search(r"\d{4}", start + "0000").group())
        end_year = 2026 if end in ("present", "now") else int(re.search(r"\d{4}", end).group())
        total += max(0, end_year - start_year)
    return float(total)


def extract_education_level(text: str) -> int:
    """Highest education level mentioned (PhD > Master > Bachelor > HS)."""
    for level in sorted(EDUCATION_KEYWORDS, reverse=True):
        if any(kw in text for kw in EDUCATION_KEYWORDS[level]):
            return level
    return 1  # default: Bachelor


def extract_gpa(text: str) -> float:
    """Find a GPA/CGPA value, normalised to a 4.0 scale."""
    m = re.search(r"(?:gpa|cgpa)[:\s]*([0-4](?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"([0-4]\.\d+)\s*/\s*4", text)
    return float(m.group(1)) if m else 3.0


def count_keywords(text: str, keywords) -> int:
    return sum(1 for kw in keywords if kw in text)


def skill_match(text: str, job_skills) -> float:
    """Percentage of required job skills present in the resume."""
    if not job_skills:
        return 60.0  # neutral default when no job skills provided
    hits = sum(1 for s in job_skills if s.lower().replace("-", " ") in text
               or s.lower() in text)
    return round(100 * hits / len(job_skills), 1)


def parse_resume(text: str, job_skills=None,
                 interview_score: float = 60, communication_score: float = 60) -> pd.DataFrame:
    """Turn resume text into a one-row feature DataFrame."""
    low = text.lower()
    features = {
        "years_experience": extract_years_experience(low),
        "education_level": extract_education_level(low),
        "skill_match_score": skill_match(low, job_skills or []),
        "interview_score": interview_score,          # not in resume
        "communication_score": communication_score,  # not in resume
        "num_certifications": min(10, count_keywords(low, CERT_KEYWORDS)),
        "num_projects": min(20, len(re.findall(r"\bproject\b", low))),
        "gpa": extract_gpa(low),
    }
    return pd.DataFrame([features])[NUMERIC_FEATURES]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features from a text resume.")
    parser.add_argument("resume", help="Path to a plain-text resume file.")
    parser.add_argument("--job-skills", default="",
                        help="Comma-separated required skills, e.g. python,sql,docker")
    parser.add_argument("--interview-score", type=float, default=60)
    parser.add_argument("--communication-score", type=float, default=60)
    parser.add_argument("--screen", action="store_true",
                        help="Also run the screening model on the parsed features.")
    args = parser.parse_args()

    text = open(args.resume, encoding="utf-8").read()
    skills = [s.strip() for s in args.job_skills.split(",") if s.strip()]
    features = parse_resume(text, skills, args.interview_score, args.communication_score)

    print("Extracted features:\n")
    for col in NUMERIC_FEATURES:
        print(f"  {col:22s} {features.loc[0, col]}")

    if args.screen:
        from predict import load_model, screen
        model = load_model()["model"]
        result = screen(model, features).iloc[0]
        marker = "✅" if result["decision"] == "Shortlist" else "❌"
        print(f"\n{marker} {result['decision']} "
              f"(shortlist probability {result['shortlist_probability']:.2f})")


if __name__ == "__main__":
    main()
