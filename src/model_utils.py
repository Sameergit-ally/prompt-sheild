"""
Shared model loading and inference helpers for Prompt Shield.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NoOpNlpEngine
from presidio_anonymizer import AnonymizerEngine
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_NAMES = ["safe", "jailbreak", "pii_leak"]
MAX_LENGTH = 64


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    probabilities: dict[str, float]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_model_dir() -> Path:
    env_dir = os.getenv("PROMPT_SHIELD_MODEL_DIR")
    if env_dir:
        return Path(env_dir)

    repo_model_dir = _repo_root() / "models" / "baseline-distilbert"
    if repo_model_dir.exists():
        return repo_model_dir

    return Path("C:/ps_models/baseline-distilbert")


@lru_cache(maxsize=1)
def load_classifier() -> tuple[Any, Any, torch.device]:
    model_dir = get_model_dir()
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Could not find the trained model directory at {model_dir}. "
            "Set PROMPT_SHIELD_MODEL_DIR to the saved DistilBERT folder."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_prompt(text: str) -> PredictionResult:
    tokenizer, model, device = load_classifier()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu().tolist()

    probability_map = {
        label: float(probabilities[index])
        for index, label in enumerate(LABEL_NAMES)
    }
    predicted_index = int(max(range(len(probabilities)), key=lambda index: probabilities[index]))
    return PredictionResult(
        label=LABEL_NAMES[predicted_index],
        confidence=float(probabilities[predicted_index]),
        probabilities=probability_map,
    )


@lru_cache(maxsize=1)
def get_analyzer() -> AnalyzerEngine:
    analyzer = AnalyzerEngine(
        nlp_engine=NoOpNlpEngine(models=[{"lang_code": "en", "model_name": "no_op"}])
    )
    analyzer.nlp_engine.load()

    api_key_patterns = [
        Pattern(name="openai_api_key", regex=r"sk-[A-Za-z0-9]{20,}", score=0.95),
        Pattern(name="aws_access_key", regex=r"AKIA[0-9A-Z]{16}", score=0.95),
        Pattern(name="github_token", regex=r"gh[pousr]_[A-Za-z0-9_]{20,}", score=0.95),
        Pattern(name="generic_api_key", regex=r"(?i)(?:api[_-]?key|secret[_-]?key|token)[:=\s]+[A-Za-z0-9_\-]{16,}", score=0.75),
    ]
    analyzer.registry.add_recognizer(PatternRecognizer(supported_entity="API_KEY", patterns=api_key_patterns))
    return analyzer


@lru_cache(maxsize=1)
def get_anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()


def analyze_pii(text: str) -> tuple[list[dict[str, Any]], str, bool]:
    analyzer = get_analyzer()
    anonymizer = get_anonymizer()

    entities = ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "API_KEY"]
    results = analyzer.analyze(text=text, entities=entities, language="en")
    redacted = anonymizer.anonymize(text=text, analyzer_results=results).text if results else text

    pii_findings = [
        {
            "entity_type": result.entity_type,
            "start": result.start,
            "end": result.end,
            "score": float(result.score),
            "text": text[result.start:result.end],
        }
        for result in results
    ]
    return pii_findings, redacted, bool(results)


def build_prediction_payload(text: str) -> dict[str, Any]:
    prediction = predict_prompt(text)
    pii_findings, redacted_text, has_pii = analyze_pii(text)

    return {
        "label": prediction.label,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "has_pii": has_pii,
        "pii_findings": pii_findings,
        "text_redacted": redacted_text,
    }