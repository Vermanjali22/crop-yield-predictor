import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import os

# Load cleaned data
df = pd.read_csv("data/crop_production_cleaned.csv")

# --- Features & Target ---
features = ["State_Name", "Crop_Year", "Season", "Crop", "Area"]
target = "Yield"

df = df[features + [target]]

# --- Encode categorical columns ---
le_state  = LabelEncoder()
le_season = LabelEncoder()
le_crop   = LabelEncoder()

df["State_Name"] = le_state.fit_transform(df["State_Name"])
df["Season"]     = le_season.fit_transform(df["Season"])
df["Crop"]       = le_crop.fit_transform(df["Crop"])

# --- Split data ---
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")

# --- Train model ---
print("\n⏳ Training model... (may take 1-2 minutes)")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
print(f"\n✅ Model trained!")
print(f"R² Score     : {r2_score(y_test, y_pred):.4f}  (closer to 1 = better)")
print(f"Mean Abs Error: {mean_absolute_error(y_test, y_pred):.4f}")

# --- Save model + encoders ---
os.makedirs("model", exist_ok=True)
pickle.dump(model,    open("model/model.pkl",    "wb"))
pickle.dump(le_state, open("model/le_state.pkl", "wb"))
pickle.dump(le_season,open("model/le_season.pkl","wb"))
pickle.dump(le_crop,  open("model/le_crop.pkl",  "wb"))

print("\n💾 Model saved to model/ folder!")