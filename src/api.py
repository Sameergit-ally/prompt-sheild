"""
FastAPI backend for Prompt Shield.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.model_utils import build_prediction_payload


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


class PiiFinding(BaseModel):
    entity_type: str
    start: int
    end: int
    score: float
    text: str


class PredictResponse(BaseModel):
    label: Literal["safe", "jailbreak", "pii_leak"]
    confidence: float
    probabilities: dict[str, float]
    has_pii: bool
    pii_findings: list[PiiFinding]
    text_redacted: str


app = FastAPI(title="Prompt Shield API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict[str, Any]:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Prompt text cannot be empty.")

    try:
        return build_prediction_payload(text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive API guard
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc