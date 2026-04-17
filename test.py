import pandas as pd
import json

# Load RAW data instead of featured
df = pd.read_csv("data/merged_train_data.csv")

# Select minimal raw columns needed
cols = [
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "dist1",
    "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo",
    "C1", "C2", "C3", "C5", "C7",
    "D1", "D2", "D15"
]

df = df[cols]

samples = df.head(3).to_dict(orient="records")

# Optional context (stateful features)
context = [
    {
        "TimeSinceLastTransaction": 3600,
        "TransactionVelocity1h": 2,
        "TransactionVelocity24h": 6
    }
    for _ in range(len(samples))
]

payload = {
    "records": samples,
    "context": context
}

with open("sample_request.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, default=float)

print("Saved RAW samples to sample_request.json")