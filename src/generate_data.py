"""
HireSense AI — synthetic applicant dataset generator.

Real hiring data is sensitive and rarely public, so we generate a realistic
labelled dataset. Each applicant's "shortlisted" label is derived from a
weighted competency score plus noise, giving the SVM a learnable but
non-trivial decision boundary.
"""
import argparse

import numpy as np
import pandas as pd

from config import (
    DATASET_PATH,
    GROUP_COLUMN,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
)


def generate(n_samples: int = 2000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Build a realistic synthetic applicant dataset."""
    rng = np.random.default_rng(seed)

    years_experience = np.clip(rng.gamma(shape=2.0, scale=2.5, size=n_samples), 0, 25)
    education_level = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.15, 0.5, 0.3, 0.05])
    skill_match_score = np.clip(rng.normal(60, 18, n_samples), 0, 100)
    interview_score = np.clip(rng.normal(62, 16, n_samples), 0, 100)
    communication_score = np.clip(rng.normal(65, 15, n_samples), 0, 100)
    num_certifications = rng.poisson(1.5, n_samples).clip(0, 10)
    num_projects = rng.poisson(3, n_samples).clip(0, 20)
    gpa = np.clip(rng.normal(3.1, 0.4, n_samples), 2.0, 4.0)

    df = pd.DataFrame({
        "years_experience": years_experience.round(1),
        "education_level": education_level,
        "skill_match_score": skill_match_score.round(1),
        "interview_score": interview_score.round(1),
        "communication_score": communication_score.round(1),
        "num_certifications": num_certifications,
        "num_projects": num_projects,
        "gpa": gpa.round(2),
    })

    # Weighted competency score — reflects what a recruiter would value.
    score = (
        0.030 * df["years_experience"] * 10      # experience matters, capped by clip
        + 0.15 * df["education_level"] * 10
        + 0.030 * df["skill_match_score"]
        + 0.028 * df["interview_score"]
        + 0.020 * df["communication_score"]
        + 0.10 * df["num_certifications"]
        + 0.06 * df["num_projects"]
        + 0.40 * df["gpa"]
    )

    # Normalise to a 0-1 probability via a logistic function, add noise,
    # then threshold. This keeps classes roughly balanced but overlapping.
    score = (score - score.mean()) / score.std()
    noise = rng.normal(0, 0.6, n_samples)
    prob = 1 / (1 + np.exp(-(score + noise)))
    df[TARGET] = (prob >= 0.5).astype(int)

    # Protected attribute for fairness auditing. Drawn independently of the
    # label so it is NOT used as a model feature — it only lets us check the
    # model's decisions for demographic parity later.
    df[GROUP_COLUMN] = rng.choice(["A", "B"], size=n_samples, p=[0.5, 0.5])

    # Guarantee column order: features first, target, then protected attribute.
    return df[NUMERIC_FEATURES + [TARGET, GROUP_COLUMN]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic applicant dataset.")
    parser.add_argument("-n", "--n-samples", type=int, default=2000)
    parser.add_argument("-s", "--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    df = generate(args.n_samples, args.seed)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)

    pos = int(df[TARGET].sum())
    print(f"✓ {len(df)} applicants written to {DATASET_PATH}")
    print(f"  Shortlisted: {pos} ({pos / len(df):.1%})  |  Rejected: {len(df) - pos}")


if __name__ == "__main__":
    main()
