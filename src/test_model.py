"""
Quick interactive test script for the trained Prompt Shield baseline model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "C:/ps_models/baseline-distilbert"
LABEL_NAMES = ["safe", "jailbreak", "pii_leak"]

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

print("Model loaded. Type a prompt to classify it (or 'quit' to exit).\n")

while True:
    text = input("Prompt: ").strip()
    if text.lower() in ("quit", "exit"):
        break
    if not text:
        continue

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    pred_idx = torch.argmax(probs).item()
    print(f"  -> Prediction: {LABEL_NAMES[pred_idx]}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"     {name}: {probs[i].item():.3f}")
    print()