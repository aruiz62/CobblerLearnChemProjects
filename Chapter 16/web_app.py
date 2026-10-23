
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# Load the trained Random Forest model
model_path = Path(__file__).parent / "housing_model.pkl"
model = joblib.load(model_path)

# Website settings
st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 California Housing Price Predictor")
st.write("Enter housing information to predict a house price.")

st.divider()

# Housing inputs
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Median Income", value=5.0)
    rooms = st.number_input("Average Rooms", value=5.0)
    population = st.number_input("Population", value=1200.0)
    latitude = st.number_input("Latitude", value=34.0)

with col2:
    age = st.number_input("House Age", value=25.0)
    bedrooms = st.number_input("Average Bedrooms", value=1.0)
    occupancy = st.number_input("Average Occupancy", value=3.0)
    longitude = st.number_input("Longitude", value=-118.0)

st.divider()

# Predict house price
if st.button("Predict House Price", type="primary"):

    data = pd.DataFrame([{
        "MedInc": income,
        "HouseAge": age,
        "AveRooms": rooms,
        "AveBedrms": bedrooms,
        "Population": population,
        "AveOccup": occupancy,
        "Latitude": latitude,
        "Longitude": longitude
    }])

    prediction = model.predict(data)[0]
    price = prediction * 100000

    st.success("Prediction completed!")
    st.metric("Estimated House Price", f"${price:,.2f}")

st.caption("Powered by Random Forest Regression")
