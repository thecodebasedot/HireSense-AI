"""
HireSense AI — data validation.

Validate an applicant DataFrame against the expected schema before scoring or
training: required columns present, numeric dtypes, and values within the
allowed ranges. Catches malformed inputs early instead of producing silently
wrong predictions.
"""
import pandas as pd

from config import FEATURE_RANGES, NUMERIC_FEATURES


class ValidationError(Exception):
    """Raised when a dataset fails validation (with a list of issues)."""

    def __init__(self, issues):
        self.issues = issues
        super().__init__(f"{len(issues)} validation issue(s): " + "; ".join(issues))


def validate(df: pd.DataFrame, raise_on_error: bool = False) -> list[str]:
    """Return a list of validation issues (empty list == valid)."""
    issues: list[str] = []

    missing = [c for c in NUMERIC_FEATURES if c not in df.columns]
    if missing:
        issues.append(f"missing columns: {missing}")
        # Can't check further without the columns.
        if raise_on_error:
            raise ValidationError(issues)
        return issues

    for col in NUMERIC_FEATURES:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            issues.append(f"'{col}' is not numeric")
            continue
        if series.isna().any():
            issues.append(f"'{col}' has {int(series.isna().sum())} missing value(s)")
        low, high = FEATURE_RANGES[col]
        out_of_range = series[(series < low) | (series > high)]
        if len(out_of_range):
            issues.append(
                f"'{col}' has {len(out_of_range)} value(s) outside [{low}, {high}]"
            )

    if raise_on_error and issues:
        raise ValidationError(issues)
    return issues


def is_valid(df: pd.DataFrame) -> bool:
    return len(validate(df)) == 0


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Validate an applicant CSV.")
    parser.add_argument("csv", help="Path to the CSV to validate.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    issues = validate(df)
    if not issues:
        print(f"✓ {args.csv} is valid ({len(df)} rows).")
    else:
        print(f"✗ {len(issues)} issue(s) in {args.csv}:")
        for i in issues:
            print(f"  - {i}")


if __name__ == "__main__":
    main()
