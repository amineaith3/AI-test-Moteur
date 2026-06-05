# %% [markdown]
# # Step D (Version 5): The Hierarchical "Router" Model
# At prediction time, we don't know if a car is Economy or Luxury.
# So we build a 3-Stage AI:
# 1. **The Classifier:** Predicts if the car is Economy (<= 300k) or Luxury (> 300k).
# 2. **The Economy Regressor:** Highly tuned for cars <= 300k.
# 3. **The Luxury Regressor:** Highly tuned for cars > 300k.

# %%
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# %%
print("Loading V5 dataset...")
df = pd.read_csv('../data/moteur_ma_preprocessed_v5.csv')

X = df.drop('Price_MAD', axis=1)
y = df['Price_MAD']

# Create Classification Target: 0 for Economy, 1 for Luxury
y_class = (y > 300000).astype(int)

cat_cols = ['Brand_Model', 'Transmission', 'Fuel', 'Location_Clean']

# --- STAGE 1: Train Classifier ---
print("Training Classifier (Economy vs Luxury Router)...")
classifier_pipeline = Pipeline([
    ('encoder', TargetEncoder(cols=cat_cols)),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])
X_tr, X_te, y_c_tr, y_c_te = train_test_split(X, y_class, test_size=0.2, random_state=42)
classifier_pipeline.fit(X_tr, y_c_tr)
class_preds = classifier_pipeline.predict(X_te)
print(f"Classifier Accuracy: {accuracy_score(y_c_te, class_preds) * 100:.2f}%")

# --- STAGE 2: Train Economy Model ---
print("\\nTraining Economy Model...")
df_economy = df[df['Price_MAD'] <= 300000]
X_eco = df_economy.drop('Price_MAD', axis=1)
y_eco = df_economy['Price_MAD']

eco_pipeline = Pipeline([
    ('encoder', TargetEncoder(cols=cat_cols)),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])
X_e_tr, X_e_te, y_e_tr, y_e_te = train_test_split(X_eco, y_eco, test_size=0.2, random_state=42)
eco_pipeline.fit(X_e_tr, y_e_tr)
eco_preds = eco_pipeline.predict(X_e_te)
print(f"Economy MAE: {mean_absolute_error(y_e_te, eco_preds):.2f} MAD")

# --- STAGE 3: Train Luxury Model ---
print("\\nTraining Luxury Model...")
df_luxury = df[df['Price_MAD'] > 300000]
X_lux = df_luxury.drop('Price_MAD', axis=1)
y_lux = df_luxury['Price_MAD']

lux_pipeline = Pipeline([
    ('encoder', TargetEncoder(cols=cat_cols)),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])
X_l_tr, X_l_te, y_l_tr, y_l_te = train_test_split(X_lux, y_lux, test_size=0.2, random_state=42)
lux_pipeline.fit(X_l_tr, y_l_tr)
lux_preds = lux_pipeline.predict(X_l_te)
print(f"Luxury MAE: {mean_absolute_error(y_l_te, lux_preds):.2f} MAD")

# --- EXPORT FINAL MASTER MODELS FOR WEB APP ---
print("\\nExporting V5 Models to app folder...")
classifier_pipeline.fit(X, y_class) # Retrain on full for production
eco_pipeline.fit(X_eco, y_eco)
lux_pipeline.fit(X_lux, y_lux)

joblib.dump(classifier_pipeline, '../app/model_classifier.pkl')
joblib.dump(eco_pipeline, '../app/model_economy.pkl')
joblib.dump(lux_pipeline, '../app/model_luxury.pkl')

# Export Metadata
metadata = {
    'Brand_Model': sorted(X['Brand_Model'].unique().tolist()),
    'Transmission': sorted(X['Transmission'].unique().tolist()),
    'Fuel': sorted(X['Fuel'].unique().tolist()),
    'Location_Clean': sorted(X['Location_Clean'].unique().tolist())
}
with open('../app/metadata.json', 'w') as f:
    json.dump(metadata, f)

print("V5 Export Complete!")
