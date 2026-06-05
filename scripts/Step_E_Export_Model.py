import pandas as pd
import joblib
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestRegressor

print("Loading V3 dataset to train the final V4 Economy Model...")
df = pd.read_csv('../data/moteur_ma_preprocessed_v3.csv')

# V4 Economy Filter
df_economy = df[df['Price_MAD'] <= 300000].copy()

X = df_economy.drop('Price_MAD', axis=1)
y = df_economy['Price_MAD']

cat_cols = ['Brand', 'CarModel', 'Transmission', 'Fuel', 'Location_Clean']

# Final Model Pipeline
final_pipeline = Pipeline([
    ('encoder', TargetEncoder(cols=cat_cols)),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

print("Training final model on 100% of the Economy data...")
final_pipeline.fit(X, y)

# Export the model
model_path = '../app/model.pkl'
joblib.dump(final_pipeline, model_path)
print(f"Model exported successfully to {model_path}")

# Export unique categorical values for the web app dropdowns
metadata = {
    'Brand': sorted(X['Brand'].unique().tolist()),
    'CarModel': sorted(X['CarModel'].unique().tolist()),
    'Transmission': sorted(X['Transmission'].unique().tolist()),
    'Fuel': sorted(X['Fuel'].unique().tolist()),
    'Location_Clean': sorted(X['Location_Clean'].unique().tolist())
}

with open('../app/metadata.json', 'w') as f:
    json.dump(metadata, f)
print("Metadata exported successfully for Streamlit dropdowns.")
