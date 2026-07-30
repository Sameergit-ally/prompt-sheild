"""
Pulls real-world jailbreak vs benign prompts from HuggingFace
(jackhhao/jailbreak-classification dataset) and saves them as a CSV.
"""

from datasets import load_dataset
import pandas as pd

print("Downloading jackhhao/jailbreak-classification...")
ds = load_dataset("jackhhao/jailbreak-classification")

train = ds["train"]
test = ds["test"]

print(f"Train rows: {len(train)}, Test rows: {len(test)}")
print("Sample row:", train[0])

rows = []
for split in [train, test]:
    for row in split:
        prompt = row["prompt"]
        label_type = row["type"]  # 'jailbreak' or 'benign'
        if label_type == "jailbreak":
            label, label_name = 1, "jailbreak"
        else:
            label, label_name = 0, "safe"
        rows.append({"prompt": prompt, "label": label, "label_name": label_name})

df = pd.DataFrame(rows)
df = df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)

print("\nClass distribution after dedup:")
print(df["label_name"].value_counts())

output_path = "data/hf_jailbreak_safe.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved {len(df)} rows to {output_path}")