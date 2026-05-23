import os
import pandas as pd
import pickle
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- Auto-train if model doesn't exist ---
@st.cache_resource
def load_model():
    if not os.path.exists("model/model.pkl"):

        df = pd.read_csv("data/crop_production_cleaned.csv").sample(50000, random_state=42)
        features = ["State_Name", "Crop_Year", "Season", "Crop", "Area"]

        le_state  = LabelEncoder()
        le_season = LabelEncoder()
        le_crop   = LabelEncoder()

        df["State_Name"] = le_state.fit_transform(df["State_Name"])
        df["Season"]     = le_season.fit_transform(df["Season"])
        df["Crop"]       = le_crop.fit_transform(df["Crop"])

        X = df[features]
        y = df["Yield"]

        model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
        model.fit(X, y)

        os.makedirs("model", exist_ok=True)
        pickle.dump(model,     open("model/model.pkl",    "wb"))
        pickle.dump(le_state,  open("model/le_state.pkl", "wb"))
        pickle.dump(le_season, open("model/le_season.pkl","wb"))
        pickle.dump(le_crop,   open("model/le_crop.pkl",  "wb"))

    model     = pickle.load(open("model/model.pkl",    "rb"))
    le_state  = pickle.load(open("model/le_state.pkl", "rb"))
    le_season = pickle.load(open("model/le_season.pkl","rb"))
    le_crop   = pickle.load(open("model/le_crop.pkl",  "rb"))
    return model, le_state, le_season, le_crop

# --- Page config ---
st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾")
st.title("🌾 Crop Yield Predictor")
st.markdown("Predict crop yield (tonnes per hectare) based on farming conditions.")

with st.spinner("⏳ Loading model... please wait"):
    model, le_state, le_season, le_crop = load_model()

# --- Input form ---
st.header("Enter Crop Details")
col1, col2 = st.columns(2)

with col1:
    state  = st.selectbox("State",  sorted(le_state.classes_))
    season = st.selectbox("Season", sorted(le_season.classes_))
    crop   = st.selectbox("Crop",   sorted(le_crop.classes_))

with col2:
    year = st.slider("Crop Year", min_value=1997, max_value=2025, value=2020)
    area = st.number_input("Area (in hectares)", min_value=0.1, value=100.0)

# --- Predict ---
if st.button("🔍 Predict Yield"):
    state_enc  = le_state.transform([state])[0]
    season_enc = le_season.transform([season])[0]
    crop_enc   = le_crop.transform([crop])[0]

    input_data = pd.DataFrame([[state_enc, year, season_enc, crop_enc, area]],
                               columns=["State_Name", "Crop_Year", "Season", "Crop", "Area"])

    prediction = model.predict(input_data)[0]
    st.success(f"🌱 Predicted Yield: **{prediction:.2f} tonnes/hectare**")

    if prediction < 1:
        st.warning("⚠️ Low yield expected. Consider soil quality and irrigation.")
    elif prediction < 5:
        st.info("📊 Average yield range. Typical for most crops.")
    else:
        st.balloons()
        st.success("🚀 High yield expected! Great conditions.")