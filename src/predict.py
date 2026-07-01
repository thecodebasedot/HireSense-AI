"""
HireSense AI — আবেদনকারী স্ক্রিনিং / প্রেডিকশন
Screen new applicants with the trained SVM model.

Usage:
    # Single applicant via flags
    python src/predict.py --years-experience 5 --education-level 2 \
        --skill-match-score 80 --interview-score 75 \
        --communication-score 70 --num-certifications 3 \
        --num-projects 6 --gpa 3.6

    # Batch scoring from a CSV (must contain the feature columns)
    python src/predict.py --csv data/new_applicants.csv
"""
import argparse

import joblib
import pandas as pd

from config import EDUCATION_LABELS, MODEL_PATH, NUMERIC_FEATURES


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python src/train.py` first."
        )
    return joblib.load(path)


def screen(model, df: pd.DataFrame) -> pd.DataFrame:
    """প্রতিটি আবেদনকারীর জন্য সিদ্ধান্ত ও সম্ভাবনা ফেরত দেয়।"""
    missing = [c for c in NUMERIC_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[NUMERIC_FEATURES]
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    out = df.copy()
    out["shortlist_probability"] = proba.round(4)
    out["decision"] = ["Shortlist" if p == 1 else "Reject" for p in pred]
    return out


def _single_from_args(args) -> pd.DataFrame:
    return pd.DataFrame([{
        "years_experience": args.years_experience,
        "education_level": args.education_level,
        "skill_match_score": args.skill_match_score,
        "interview_score": args.interview_score,
        "communication_score": args.communication_score,
        "num_certifications": args.num_certifications,
        "num_projects": args.num_projects,
        "gpa": args.gpa,
    }])


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen applicants with the SVM model.")
    parser.add_argument("--csv", help="Path to a CSV of applicants for batch scoring.")
    parser.add_argument("--years-experience", type=float, default=0)
    parser.add_argument("--education-level", type=int, default=1,
                        help="0=HS, 1=Bachelor, 2=Master, 3=PhD")
    parser.add_argument("--skill-match-score", type=float, default=50)
    parser.add_argument("--interview-score", type=float, default=50)
    parser.add_argument("--communication-score", type=float, default=50)
    parser.add_argument("--num-certifications", type=int, default=0)
    parser.add_argument("--num-projects", type=int, default=0)
    parser.add_argument("--gpa", type=float, default=3.0)
    args = parser.parse_args()

    bundle = load_model()
    model = bundle["model"]

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        df = _single_from_args(args)

    result = screen(model, df)

    print(f"Model test accuracy: {bundle.get('test_accuracy', float('nan')):.3f} "
          f"| ROC-AUC: {bundle.get('test_roc_auc', float('nan')):.3f}\n")

    for _, row in result.iterrows():
        edu = EDUCATION_LABELS.get(int(row["education_level"]), row["education_level"])
        marker = "✅" if row["decision"] == "Shortlist" else "❌"
        print(f"{marker} {row['decision']:10s} "
              f"(P={row['shortlist_probability']:.2f})  "
              f"exp={row['years_experience']}y, edu={edu}, "
              f"skill={row['skill_match_score']}, interview={row['interview_score']}")

    if args.csv:
        out_path = args.csv.rsplit(".", 1)[0] + "_scored.csv"
        result.to_csv(out_path, index=False)
        print(f"\n✓ Scored results written to {out_path}")


if __name__ == "__main__":
    main()
