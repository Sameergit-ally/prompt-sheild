"""
Merges all data sources into one final dataset for training.
"""

import pandas as pd

files = [
    "data/dataset.csv",
    "data/hf_jailbreak_safe.csv",
    "data/synthetic_pii.csv",
    "data/soft_jailbreak.csv",
]

dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        print(f"Loaded {f}: {len(df)} rows")
        dfs.append(df[["prompt", "label", "label_name"]])
    except FileNotFoundError:
        print(f"WARNING: {f} not found, skipping.")

combined = pd.concat(dfs, ignore_index=True)
before = len(combined)
combined = combined.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
after = len(combined)
print(f"\nTotal rows before dedup: {before}, after dedup: {after}")

print("\nFinal class distribution:")
print(combined["label_name"].value_counts())

output_path = "data/final_dataset.csv"
combined.to_csv(output_path, index=False)
print(f"\nSaved merged dataset to {output_path}")
