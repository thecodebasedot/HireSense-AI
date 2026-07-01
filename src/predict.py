"""
HireSense AI — screen new applicants with the trained SVM model.

Usage:
    # Single applicant via flags
    python src/predict.py --years-experience 5 --education-level 2 \
        --skill-match-score 80 --interview-score 75 \
        --communication-score 70 --num-certifications 3 \
        --num-projects 6 --gpa 3.6

    # Batch scoring from a CSV (must contain the feature columns)
    python src/predict.py --csv data/new_applicants.csv

    # Rank the top 3 candidates in a CSV
    python src/predict.py --csv data/sample_applicants.csv --top 3

    # Apply a job profile (custom threshold + hard requirements)
    python src/predict.py --csv data/sample_applicants.csv --profile senior_engineer

    # Use a custom decision threshold
    python src/predict.py --csv data/sample_applicants.csv --threshold 0.7
"""
import argparse

import joblib
import pandas as pd

from config import DEFAULT_THRESHOLD, EDUCATION_LABELS, MODEL_PATH, NUMERIC_FEATURES


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python src/train.py` first."
        )
    return joblib.load(path)


def screen(model, df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD,
           profile=None) -> pd.DataFrame:
    """Return a decision and probability for each applicant.

    threshold : shortlist if probability >= threshold (default 0.5).
    profile   : optional job profile (see job_profiles). When given, its
                threshold overrides `threshold` and its hard requirements can
                veto a shortlist regardless of probability.
    """
    missing = [c for c in NUMERIC_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    if profile is not None:
        threshold = profile.get("threshold", threshold)

    X = df[NUMERIC_FEATURES]
    proba = model.predict_proba(X)[:, 1]

    out = df.copy()
    out["shortlist_probability"] = proba.round(4)
    decisions = proba >= threshold

    if profile is not None:
        # Apply hard requirements: any unmet minimum vetoes the shortlist.
        from job_profiles import meets_requirements  # local import avoids cycle
        met = df.apply(lambda r: meets_requirements(r, profile), axis=1)
        decisions = decisions & met.values

    out["decision"] = ["Shortlist" if d else "Reject" for d in decisions]
    return out


def rank(model, df: pd.DataFrame, top: int | None = None, **kwargs) -> pd.DataFrame:
    """Score and rank applicants by shortlist probability (highest first)."""
    scored = screen(model, df, **kwargs).sort_values(
        "shortlist_probability", ascending=False
    ).reset_index(drop=True)
    scored.insert(0, "rank", scored.index + 1)
    return scored.head(top) if top else scored


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
    parser.add_argument("--top", type=int, help="Rank and show the top N candidates.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Decision threshold on shortlist probability (default 0.5).")
    parser.add_argument("--profile", help="Job profile name (see profiles/).")
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

    profile = None
    if args.profile:
        from job_profiles import load_profile
        profile = load_profile(args.profile)

    df = pd.read_csv(args.csv) if args.csv else _single_from_args(args)

    print(f"Model v{bundle.get('version', '?')} ({bundle.get('algorithm', 'SVM')}) "
          f"| accuracy {bundle.get('test_accuracy', float('nan')):.3f} "
          f"| ROC-AUC {bundle.get('test_roc_auc', float('nan')):.3f}")
    if profile:
        print(f"Job profile: {args.profile} (threshold={profile.get('threshold')})")
    elif args.threshold != DEFAULT_THRESHOLD:
        print(f"Threshold: {args.threshold}")
    print()

    if args.top:
        result = rank(model, df, top=args.top, threshold=args.threshold, profile=profile)
        for _, row in result.iterrows():
            marker = "✅" if row["decision"] == "Shortlist" else "❌"
            print(f"#{int(row['rank']):<2d} {marker} {row['decision']:10s} "
                  f"(P={row['shortlist_probability']:.2f})  "
                  f"exp={row['years_experience']}y, "
                  f"skill={row['skill_match_score']}, interview={row['interview_score']}")
    else:
        result = screen(model, df, threshold=args.threshold, profile=profile)
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
