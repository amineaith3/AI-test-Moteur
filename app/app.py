import streamlit as st
import pandas as pd
import joblib
import json
import os

st.set_page_config(page_title="Moteur.ma AI App V5", page_icon="🚗", layout="centered")

@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    classifier = joblib.load(os.path.join(base_dir, 'model_classifier.pkl'))
    economy = joblib.load(os.path.join(base_dir, 'model_economy.pkl'))
    luxury = joblib.load(os.path.join(base_dir, 'model_luxury.pkl'))
    return classifier, economy, luxury

@st.cache_data
def load_metadata():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, 'metadata.json'), 'r') as f:
        return json.load(f)

classifier, eco_model, lux_model = load_models()
metadata = load_metadata()

st.title("🏎️ AI Car Price Estimator V5")
st.markdown("""
**Hierarchical Routing Engine Active.** 
This AI uses a Classification network to detect if your car belongs to the Economy or Luxury tier, 
then mathematically routes it to a dedicated Regressor algorithm for maximum accuracy.
""")

st.header("Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    brand_model = st.selectbox("Brand & Model", metadata['Brand_Model'])
    year = st.slider("Year of Manufacture", 1990, 2025, 2015)

with col2:
    fuel = st.selectbox("Fuel Type", metadata['Fuel'])
    transmission = st.selectbox("Transmission", metadata['Transmission'])
    location = st.selectbox("City / Location", metadata['Location_Clean'])

mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=100000, step=5000)

if st.button("Predict Fair Market Price", type="primary"):
    input_data = pd.DataFrame({
        'Brand_Model': [brand_model],
        'Year': [year],
        'Mileage_km': [mileage],
        'Transmission': [transmission],
        'Fuel': [fuel],
        'Location_Clean': [location]
    })
    
    with st.spinner("Classifying market tier..."):
        is_luxury = classifier.predict(input_data)[0]
    
    with st.spinner("Routing to dedicated mathematical regressor..."):
        if is_luxury == 1:
            tier = "Luxury Tier (>300k MAD)"
            prediction = lux_model.predict(input_data)[0]
            error_margin = "± 45,000 MAD"
        else:
            tier = "Economy Tier (≤300k MAD)"
            prediction = eco_model.predict(input_data)[0]
            error_margin = "± 16,000 MAD"
    
    st.success("Analysis Complete!")
    st.markdown(f"### 🏷️ Estimated Market Value: **{prediction:,.0f} MAD**")
    st.caption(f"Market Identified: {tier} | Expected Margin of Error: {error_margin}")
