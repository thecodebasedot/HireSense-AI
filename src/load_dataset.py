"""
HireSense AI — real dataset integration.

Ingest a real hiring CSV whose column names differ from HireSense's schema,
map them to the expected features, validate, and write a standardised
`data/applicants.csv` ready for `python src/train.py`.

A mapping is a JSON file: {"your_column": "hiresense_feature", ...}. Only the
listed columns are renamed; the target column is mapped too. Missing optional
features can be filled with a default.

    # 1) See the expected schema and a mapping template
    python src/load_dataset.py --template

    # 2) Ingest a real CSV using a mapping
    python src/load_dataset.py raw_hiring.csv --mapping mapping.json

    # 3) Then train on it
    python src/train.py
"""
import argparse
import json

import pandas as pd

from config import DATASET_PATH, EDUCATION_LABELS, NUMERIC_FEATURES, TARGET
from validation import validate

# Optional features that may be absent in real data, with neutral defaults.
DEFAULTS = {
    "interview_score": 60,
    "communication_score": 60,
    "num_certifications": 0,
    "num_projects": 0,
}

# Common textual education values mapped to the ordinal scale.
EDUCATION_TEXT = {
    "high school": 0, "hs": 0, "secondary": 0, "diploma": 0,
    "bachelor": 1, "bachelors": 1, "bsc": 1, "b.sc": 1, "undergraduate": 1,
    "master": 2, "masters": 2, "msc": 2, "m.sc": 2, "mba": 2, "graduate": 2,
    "phd": 3, "doctorate": 3, "ph.d": 3,
}


def _template() -> dict:
    """A mapping template: hiresense feature -> (your column name here)."""
    schema = {f: f"<your column for {f}>" for f in NUMERIC_FEATURES}
    schema[TARGET] = "<your column for shortlisted (0/1)>"
    return schema


def normalise_education(series: pd.Series) -> pd.Series:
    """Convert textual education labels to the 0–3 ordinal scale if needed."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().map(EDUCATION_TEXT)


def ingest(csv_path: str, mapping: dict, fill_defaults: bool = True) -> pd.DataFrame:
    """Load a raw CSV and return a standardised HireSense DataFrame."""
    raw = pd.read_csv(csv_path)

    # Invert mapping (your_col -> hiresense) if given the other way is allowed:
    # accept either {hiresense: your_col} or {your_col: hiresense}.
    rename = {}
    targets = set(NUMERIC_FEATURES) | {TARGET}
    for a, b in mapping.items():
        if a in targets and b in raw.columns:      # {hiresense: your_col}
            rename[b] = a
        elif b in targets and a in raw.columns:     # {your_col: hiresense}
            rename[a] = b
    df = raw.rename(columns=rename)

    if "education_level" in df.columns:
        df["education_level"] = normalise_education(df["education_level"])

    if fill_defaults:
        for feat, default in DEFAULTS.items():
            if feat not in df.columns:
                df[feat] = default

    keep = [c for c in NUMERIC_FEATURES + [TARGET] if c in df.columns]
    df = df[keep]

    issues = validate(df)
    if issues:
        print("⚠️  Validation warnings:")
        for i in issues:
            print(f"   - {i}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a real hiring dataset.")
    parser.add_argument("csv", nargs="?", help="Path to the raw CSV.")
    parser.add_argument("--mapping", help="JSON column mapping file.")
    parser.add_argument("--template", action="store_true",
                        help="Print a mapping template and exit.")
    parser.add_argument("--out", default=str(DATASET_PATH),
                        help="Where to write the standardised CSV.")
    args = parser.parse_args()

    if args.template or not args.csv:
        print("Expected features:")
        for f in NUMERIC_FEATURES:
            extra = f"  (education: {EDUCATION_LABELS})" if f == "education_level" else ""
            print(f"  - {f}{extra}")
        print(f"  - {TARGET}  (0 = reject, 1 = shortlist)\n")
        print("Mapping template (save as mapping.json):")
        print(json.dumps(_template(), indent=2))
        return

    mapping = json.loads(open(args.mapping).read()) if args.mapping else {}
    df = ingest(args.csv, mapping)

    if TARGET not in df.columns:
        print(f"\n✗ No target column '{TARGET}' after mapping — cannot train on this.")
    df.to_csv(args.out, index=False)
    print(f"\n✓ Standardised dataset written to {args.out} "
          f"({len(df)} rows, {len(df.columns)} columns)")
    print("  Now run: python src/train.py")


if __name__ == "__main__":
    main()
