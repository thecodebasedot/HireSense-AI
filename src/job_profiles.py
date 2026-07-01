"""
HireSense AI — job profiles.

A job profile tailors screening to a specific role. It defines:
  * threshold      — decision cutoff on shortlist probability
  * requirements   — hard minimums (e.g. min years_experience) that veto a
                     shortlist if unmet, regardless of the model's probability

Profiles are JSON files in the profiles/ directory. Load one by name (the
file stem), e.g. `load_profile("senior_engineer")`.
"""
import json

from config import DEFAULT_THRESHOLD, PROFILES_DIR


def load_profile(name: str) -> dict:
    """Load a job profile by name (filename without .json)."""
    path = PROFILES_DIR / f"{name}.json"
    if not path.exists():
        available = ", ".join(p.stem for p in PROFILES_DIR.glob("*.json")) or "none"
        raise FileNotFoundError(
            f"Job profile '{name}' not found. Available: {available}"
        )
    profile = json.loads(path.read_text())
    profile.setdefault("threshold", DEFAULT_THRESHOLD)
    profile.setdefault("requirements", {})
    return profile


def meets_requirements(applicant, profile: dict) -> bool:
    """True if the applicant satisfies every hard minimum in the profile."""
    for feature, minimum in profile.get("requirements", {}).items():
        if applicant[feature] < minimum:
            return False
    return True


def list_profiles() -> list[str]:
    """Names of all available profiles."""
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))
