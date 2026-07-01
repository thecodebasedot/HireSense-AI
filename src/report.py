"""
HireSense AI — per-applicant PDF report.

Generates a one-page PDF summarising an applicant's screening result: the
decision, shortlist probability, feature values, and a per-feature
explanation of why. Rendered with matplotlib's PDF backend (no extra deps).

Usage:
    python src/report.py --years-experience 6 --education-level 2 \
        --skill-match-score 85 --interview-score 80 \
        --communication-score 78 --num-certifications 3 \
        --num-projects 8 --gpa 3.7 --name "Jane Doe"
"""
import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from config import EDUCATION_LABELS, NUMERIC_FEATURES, REPORTS_DIR
from explain import explain_instance
from predict import load_model, screen


def build_report(model, applicant: pd.DataFrame, name: str, out_path) -> None:
    result = screen(model, applicant).iloc[0]
    prob = float(result["shortlist_probability"])
    decision = result["decision"]
    contrib = explain_instance(model, applicant)

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.suptitle("HireSense AI — Applicant Screening Report", fontsize=16, fontweight="bold", y=0.97)

    # Header block
    ax_head = fig.add_axes([0.08, 0.80, 0.84, 0.12])
    ax_head.axis("off")
    color = "#2a9d8f" if decision == "Shortlist" else "#e76f51"
    ax_head.text(0, 0.8, f"Candidate: {name}", fontsize=13, fontweight="bold")
    ax_head.text(0, 0.45, f"Decision: {decision}", fontsize=13, color=color, fontweight="bold")
    ax_head.text(0, 0.1, f"Shortlist probability: {prob:.1%}", fontsize=12)

    # Feature table
    ax_tbl = fig.add_axes([0.08, 0.50, 0.84, 0.26])
    ax_tbl.axis("off")
    ax_tbl.set_title("Applicant features", loc="left", fontsize=12, fontweight="bold")
    rows = []
    for f in NUMERIC_FEATURES:
        val = applicant.loc[0, f]
        if f == "education_level":
            val = EDUCATION_LABELS.get(int(val), val)
        rows.append([f, str(val)])
    tbl = ax_tbl.table(cellText=rows, colLabels=["Feature", "Value"],
                       cellLoc="left", loc="upper left", colWidths=[0.6, 0.4])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)

    # Explanation bar chart
    ax_exp = fig.add_axes([0.12, 0.08, 0.78, 0.34])
    c = contrib.sort_values("contribution")
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in c["contribution"]]
    ax_exp.barh(c["feature"], c["contribution"], color=colors)
    ax_exp.axvline(0, color="gray", linewidth=0.8)
    ax_exp.set_title("Why? Per-feature contribution to shortlist probability",
                     fontsize=12, fontweight="bold")
    ax_exp.set_xlabel("← lowers        contribution        raises →")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PDF screening report.")
    parser.add_argument("--name", default="Applicant")
    parser.add_argument("--years-experience", type=float, default=5)
    parser.add_argument("--education-level", type=int, default=1)
    parser.add_argument("--skill-match-score", type=float, default=60)
    parser.add_argument("--interview-score", type=float, default=60)
    parser.add_argument("--communication-score", type=float, default=60)
    parser.add_argument("--num-certifications", type=int, default=1)
    parser.add_argument("--num-projects", type=int, default=3)
    parser.add_argument("--gpa", type=float, default=3.1)
    args = parser.parse_args()

    applicant = pd.DataFrame([{
        "years_experience": args.years_experience,
        "education_level": args.education_level,
        "skill_match_score": args.skill_match_score,
        "interview_score": args.interview_score,
        "communication_score": args.communication_score,
        "num_certifications": args.num_certifications,
        "num_projects": args.num_projects,
        "gpa": args.gpa,
    }])[NUMERIC_FEATURES]

    model = load_model()["model"]
    safe_name = args.name.replace(" ", "_").lower()
    out_path = REPORTS_DIR / f"report_{safe_name}.pdf"
    build_report(model, applicant, args.name, out_path)
    print(f"✓ Report saved to {out_path}")


if __name__ == "__main__":
    main()
