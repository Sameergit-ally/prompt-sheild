"""
Baseline classifier training script for Prompt Shield.
Fine-tunes DistilBERT on the merged dataset (data/final_dataset.csv)
to classify prompts as: 0 = safe, 1 = jailbreak, 2 = pii_leak.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/final_dataset.csv"
OUTPUT_DIR = "C:/ps_models/baseline-distilbert"
LABEL_NAMES = ["safe", "jailbreak", "pii_leak"]

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows")
print(df["label_name"].value_counts())

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

train_ds = Dataset.from_pandas(train_df[["prompt", "label"]].reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df[["prompt", "label"]].reset_index(drop=True))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["prompt"], truncation=True, padding="max_length", max_length=64)

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)

train_ds = train_ds.rename_column("label", "labels")
val_ds = val_ds.rename_column("label", "labels")
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABEL_NAMES)
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# NOTE: save_strategy="no" avoids mid-training checkpoint saves entirely,
# which sidesteps the Windows/OneDrive safetensors file-lock issue.
# We save the final model manually at the end instead.
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=8,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

metrics = trainer.evaluate()
print("\nFinal validation metrics:")
print(metrics)

preds_output = trainer.predict(val_ds)
preds = np.argmax(preds_output.predictions, axis=-1)
labels = preds_output.label_ids
cm = confusion_matrix(labels, preds)
print("\nConfusion matrix (rows=actual, cols=predicted):")
print("Labels order:", LABEL_NAMES)
print(cm)

# Save manually with safe_serialization=False to avoid the safetensors
# file-lock error that can happen inside OneDrive-synced folders on Windows.
model.save_pretrained(OUTPUT_DIR, safe_serialization=False)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to {OUTPUT_DIR}")