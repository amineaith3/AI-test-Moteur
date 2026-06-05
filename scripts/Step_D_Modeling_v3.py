# %% [markdown]
# # Step D (Version 3): Modeling with Target Encoding
# We replace One-Hot Encoding (Booleans) with Target Encoding.
# Target Encoding replaces a city like "Casablanca" with the *average mathematical price* of all cars in Casablanca.
# This explicitly teaches the model that "Casablanca > El Jadida" or "Mercedes > Dacia".
# This drastically reduces the number of columns from 138 back down to 7, supercharging the AI.

# %%
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
# pyrefly: ignore [missing-import]
from category_encoders import TargetEncoder

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# %%
print("Loading V3 dataset...")
df = pd.read_csv('../data/moteur_ma_preprocessed_v3.csv')

X = df.drop('Price_MAD', axis=1)
y = df['Price_MAD']

# Identify categorical columns for the Target Encoder
cat_cols = ['Brand', 'CarModel', 'Transmission', 'Fuel', 'Location_Clean']

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=42)
}

# The Pipeline now mathematically encodes the strings based on their average price BEFORE scaling
pipelines = {}
for name, model in models.items():
    pipelines[name] = Pipeline([
        ('encoder', TargetEncoder(cols=cat_cols)),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

# %%
results = []
print("Training V3 Models with Target Encoding...")

for name, pipeline in pipelines.items():
    print(f"Evaluating {name}...")
    
    # 5-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(pipeline, X, y, cv=kf, scoring='r2').mean()
    cv_mae = -cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_absolute_error').mean()
    
    results.append({"Model": name, "Strategy": "5-Fold CV", "R2_Score": round(cv_r2, 4), "MAE": round(cv_mae, 2)})
    
    # Train-Val-Test (70-15-15)
    X_tr_temp, X_val_test, y_tr_temp, y_val_test = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_val_test, y_val_test, test_size=0.50, random_state=42)
    
    pipeline.fit(X_tr_temp, y_tr_temp)
    results.append({
        "Model": name, "Strategy": "Train-Val-Test (70-15-15)",
        "R2_Score": round(r2_score(y_te, pipeline.predict(X_te)), 4),
        "MAE": round(mean_absolute_error(y_te, pipeline.predict(X_te)), 2)
    })

# %%
results_df = pd.DataFrame(results)
print("\\n================ V3 EXPERIMENT SCOREBOARD ================")
print(results_df.to_string(index=False))

results_df.to_csv('../data/modeling_results_v3.csv', index=False)
print("\\nModeling V3 complete. Scoreboard saved to data/modeling_results_v3.csv")
