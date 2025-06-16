import pandas as pd

# Load individual data
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels: 0 = fake, 1 = real
fake["subject"] = 0
real["subject"] = 1

# Combine
df = pd.concat([fake, real], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to news.csv
df.to_csv("news.csv", index=False)
print("news.csv saved successfully!")