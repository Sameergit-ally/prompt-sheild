"""
Generates synthetic PII-containing prompts using Faker.
"""

import random
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

NUM_EXAMPLES = 650

templates = [
    "My email is {PII}, can you help me draft a reply to my boss?",
    "Here's my email {PII} - please use it to sign up for the newsletter.",
    "Can you send the report to {PII} once it's ready?",
    "My phone number is {PII}, save it for the contact form please.",
    "Please call me at {PII} regarding this order.",
    "Use {PII} as the test number for this SMS API integration.",
    "My credit card number is {PII}, check if this format is valid.",
    "Here's my card {PII} for the checkout flow, does this look right?",
    "My bank account number is {PII}, format this for the report.",
    "This is my API key {PII}, can you help debug why my script is failing?",
    "My AWS secret key is {PII}, why is my S3 upload not working?",
    "Here's my GitHub token {PII}, please help fix this push error.",
    "My home address is {PII}, can you write directions from here to the airport?",
    "Please ship the package to {PII}.",
    "My full name is {PII}, use it to fill out this form.",
    "Here's my password: {PII}, is it strong enough?",
    "My database password is {PII} for this local connection string.",
    "My passport number is {PII}, check if this visa form is correct.",
    "Employee ID and details: {PII}, please onboard me into the system.",
    "My insurance policy number is {PII}, can you summarize this clause?",
]

def random_api_key():
    return "sk-" + "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=32))

def random_aws_key():
    return "AKIA" + "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=16))

def random_github_token():
    return "ghp_" + "".join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=36))

def random_password():
    words = ["Summer", "Winter", "Dragon", "Falcon", "Shadow", "Cosmic", "Silver", "Golden"]
    return f"{random.choice(words)}{random.randint(100,9999)}!"

pii_generators = [
    lambda: fake.email(),
    lambda: fake.phone_number(),
    lambda: fake.credit_card_number(),
    lambda: fake.bban(),
    lambda: random_api_key(),
    lambda: random_aws_key(),
    lambda: random_github_token(),
    lambda: fake.address().replace("\n", ", "),
    lambda: fake.name(),
    lambda: random_password(),
    lambda: fake.passport_number() if hasattr(fake, "passport_number") else fake.bothify("??######"),
]

rows = []
for _ in range(NUM_EXAMPLES):
    template = random.choice(templates)
    pii_value = random.choice(pii_generators)()
    prompt = template.replace("{PII}", str(pii_value))
    rows.append({"prompt": prompt, "label": 2, "label_name": "pii_leak"})

df = pd.DataFrame(rows)
df = df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)

print(f"Generated {len(df)} unique PII prompts")
print(df.head(5)["prompt"].to_string())

output_path = "data/synthetic_pii.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")