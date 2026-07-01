"""
HireSense AI — REST API (FastAPI).

Serve the trained SVM model over HTTP.

Run:
    uvicorn app_api:app --reload
    # then open http://127.0.0.1:8000/docs for interactive docs

Endpoints:
    GET  /health          -> service + model status
    POST /predict         -> screen a single applicant
    POST /predict/batch   -> screen a list of applicants
"""
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import EDUCATION_LABELS, NUMERIC_FEATURES  # noqa: E402
from predict import load_model, screen  # noqa: E402

app = FastAPI(
    title="HireSense AI",
    description="SVM-based job applicant screening API.",
    version="1.0.0",
)

# Load the model once at startup; reused across requests.
try:
    _BUNDLE = load_model()
    _MODEL = _BUNDLE["model"]
except FileNotFoundError:
    _BUNDLE, _MODEL = None, None


class Applicant(BaseModel):
    """One applicant's features. Field names match the model schema."""
    years_experience: float = Field(..., ge=0, le=50, examples=[6])
    education_level: int = Field(..., ge=0, le=3, description="0=HS, 1=Bachelor, 2=Master, 3=PhD", examples=[2])
    skill_match_score: float = Field(..., ge=0, le=100, examples=[85])
    interview_score: float = Field(..., ge=0, le=100, examples=[80])
    communication_score: float = Field(..., ge=0, le=100, examples=[78])
    num_certifications: int = Field(..., ge=0, le=50, examples=[3])
    num_projects: int = Field(..., ge=0, le=100, examples=[8])
    gpa: float = Field(..., ge=0, le=4, examples=[3.7])


class ScreenResult(BaseModel):
    decision: str
    shortlist_probability: float
    education: str


def _require_model():
    if _MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python src/train.py` first.",
        )


def _score(applicants: List[Applicant]) -> List[ScreenResult]:
    df = pd.DataFrame([a.model_dump() for a in applicants])[NUMERIC_FEATURES]
    scored = screen(_MODEL, df)
    return [
        ScreenResult(
            decision=row["decision"],
            shortlist_probability=float(row["shortlist_probability"]),
            education=EDUCATION_LABELS.get(int(row["education_level"]), "Unknown"),
        )
        for _, row in scored.iterrows()
    ]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _MODEL is not None,
        "test_accuracy": _BUNDLE.get("test_accuracy") if _BUNDLE else None,
        "test_roc_auc": _BUNDLE.get("test_roc_auc") if _BUNDLE else None,
    }


@app.post("/predict", response_model=ScreenResult)
def predict(applicant: Applicant):
    _require_model()
    return _score([applicant])[0]


@app.post("/predict/batch", response_model=List[ScreenResult])
def predict_batch(applicants: List[Applicant]):
    _require_model()
    if not applicants:
        raise HTTPException(status_code=400, detail="Empty applicant list.")
    return _score(applicants)
