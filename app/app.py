import streamlit as st
import pandas as pd
import joblib
import json

# Set up page layout
st.set_page_config(page_title="Moteur.ma Price Predictor", page_icon="🚗", layout="centered")

import os

# Load the trained model and metadata
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return joblib.load(os.path.join(base_dir, 'model.pkl'))

@st.cache_data
def load_metadata():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, 'metadata.json'), 'r') as f:
        return json.load(f)

model = load_model()
metadata = load_metadata()

# UI Styling
st.title("🚗 AI Car Price Estimator (Morocco)")
st.markdown("""
Welcome to the Moroccan Second-Hand Car Price Predictor! 
This AI evaluates cars specifically in the "Economy" range (under 300,000 MAD) using an algorithm trained on **10,000+ real listings** from moteur.ma.
""")

st.header("Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", metadata['Brand'])
    model_dropdown = st.selectbox("Car Model", metadata['CarModel'])
    year = st.slider("Year of Manufacture", 1990, 2025, 2015)

with col2:
    fuel = st.selectbox("Fuel Type", metadata['Fuel'])
    transmission = st.selectbox("Transmission", metadata['Transmission'])
    location = st.selectbox("City / Location", metadata['Location_Clean'])

mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=100000, step=5000)

if st.button("Predict Fair Market Price", type="primary"):
    # Create dataframe for prediction
    input_data = pd.DataFrame({
        'Brand': [brand],
        'CarModel': [model_dropdown],
        'Year': [year],
        'Mileage_km': [mileage],
        'Transmission': [transmission],
        'Fuel': [fuel],
        'Location_Clean': [location]
    })
    
    # Predict
    with st.spinner("Calculating mathematical value..."):
        prediction = model.predict(input_data)[0]
    
    st.success("Analysis Complete!")
    st.markdown(f"### 🏷️ Estimated Market Value: **{prediction:,.0f} MAD**")
    st.caption("Margin of Error: ± 16,390 MAD")
    st.info("💡 **Insight**: This price is optimized for the Moroccan market based on regional devaluation and mileage depreciation.")
