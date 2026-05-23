import pandas as pd

# Load the dataset
df = pd.read_csv("data/crop_production.csv")

# 1. See the first 5 rows
print("=== First 5 rows ===")
print(df.head())

# 2. Check shape (rows, columns)
print("\n=== Shape ===")
print(df.shape)

# 3. Check column names
print("\n=== Columns ===")
print(df.columns.tolist())

# 4. Check data types + missing values
print("\n=== Info ===")
print(df.info())

# 5. Check missing values count
print("\n=== Missing Values ===")
print(df.isnull().sum())

# 6. Basic statistics
print("\n=== Basic Stats ===")
print(df.describe())