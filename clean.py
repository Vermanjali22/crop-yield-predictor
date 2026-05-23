import pandas as pd

# Load data
df = pd.read_csv("data/crop_production.csv")

# Step 1 - Drop rows where Production is missing
df = df.dropna(subset=["Production"])

# Step 2 - Remove rows where Area is 0 (avoid division by zero)
df = df[df["Area"] > 0]

# Step 3 - Create Yield column (this is what we'll predict)
df["Yield"] = df["Production"] / df["Area"]

# Step 4 - Clean up Season column (remove extra spaces)
df["Season"] = df["Season"].str.strip()

# Step 5 - Drop outliers (top 1% of Yield — unrealistic values)
upper_limit = df["Yield"].quantile(0.99)
df = df[df["Yield"] <= upper_limit]

# Step 6 - Save cleaned data
df.to_csv("data/crop_production_cleaned.csv", index=False)

print("✅ Cleaning done!")
print(f"Rows remaining: {len(df)}")
print(f"\nSeasons available:\n{df['Season'].unique()}")
print(f"\nTop 10 crops:\n{df['Crop'].value_counts().head(10)}")
print(f"\nYield stats:\n{df['Yield'].describe()}")