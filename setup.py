
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

if not os.path.exists("model/model.pkl"):
    print("Training model...")
    df = pd.read_csv("data/crop_production_cleaned.csv")
    features = ["State_Name", "Crop_Year", "Season", "Crop", "Area"]
    
    le_state  = LabelEncoder()
    le_season = LabelEncoder()
    le_crop   = LabelEncoder()
    
    df["State_Name"] = le_state.fit_transform(df["State_Name"])
    df["Season"]     = le_season.fit_transform(df["Season"])
    df["Crop"]       = le_crop.fit_transform(df["Crop"])
    
    X = df[features]
    y = df["Yield"]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    os.makedirs("model", exist_ok=True)
    pickle.dump(model,     open("model/model.pkl",    "wb"))
    pickle.dump(le_state,  open("model/le_state.pkl", "wb"))
    pickle.dump(le_season, open("model/le_season.pkl","wb"))
    pickle.dump(le_crop,   open("model/le_crop.pkl",  "wb"))
    print("Model saved!")