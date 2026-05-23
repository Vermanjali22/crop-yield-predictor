import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾")
st.title("🌾 Crop Yield Predictor")

# Debug info
st.write("Files in data/:", os.listdir("data") if os.path.exists("data") else "data folder not found")

@st.cache_resource
def load_model():
    df = pd.read_csv("data/crop_production_cleaned.csv").sample(50000, random_state=42)
    features = ["State_Name", "Crop_Year", "Season", "Crop", "Area"]

    le_state  = LabelEncoder()
    le_season = LabelEncoder()
    le_crop   = LabelEncoder()

    df["State_Name"] = le_state.fit_transform(df["State_Name"])
    df["Season"]     = le_season.fit_transform(df["Season"])
    df["Crop"]       = le_crop.fit_transform(df["Crop"])

    model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
    model.fit(df[features], df["Yield"])

    return model, le_state, le_season, le_crop

try:
    with st.spinner("⏳ Loading model... please wait"):
        model, le_state, le_season, le_crop = load_model()
    st.success("✅ Model loaded!")
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()