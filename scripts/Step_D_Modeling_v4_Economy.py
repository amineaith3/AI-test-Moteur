# %% [markdown]
# # Step D (Version 4): Economy vs Luxury Separation
# Evaluating the model EXCLUSIVELY on "Economy" cars (prices under 300,000 MAD).
# This prevents extreme errors from million-dirham luxury cars from skewing the average error (MAE)
# and destroying the model's accuracy on everyday cars.

# %%
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from category_encoders import TargetEncoder

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# %%
print("Loading V3 dataset...")
df = pd.read_csv('../data/moteur_ma_preprocessed_v3.csv')

# --- THE V4 MAGIC: Filter out the ultra-rich ---
# Keep only cars that cost 300,000 MAD or less
df_economy = df[df['Price_MAD'] <= 300000].copy()

print(f"Total cars originally: {len(df)}")
print(f"Economy cars (<= 300k): {len(df_economy)}")

X = df_economy.drop('Price_MAD', axis=1)
y = df_economy['Price_MAD']

cat_cols = ['Brand', 'CarModel', 'Transmission', 'Fuel', 'Location_Clean']

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=42)
}

pipelines = {}
for name, model in models.items():
    pipelines[name] = Pipeline([
        ('encoder', TargetEncoder(cols=cat_cols)),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

# %%
results = []
print("\\nTraining V4 ECONOMY Models with Target Encoding...")

for name, pipeline in pipelines.items():
    print(f"Evaluating {name}...")
    
    # 5-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(pipeline, X, y, cv=kf, scoring='r2').mean()
    cv_mae = -cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_absolute_error').mean()
    
    results.append({
        "Model": name, 
        "Strategy": "5-Fold CV", 
        "R2_Score": round(cv_r2, 4), 
        "MAE": round(cv_mae, 2)
    })
    
    # Train-Val-Test (70-15-15)
    X_tr_temp, X_val_test, y_tr_temp, y_val_test = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_val_test, y_val_test, test_size=0.50, random_state=42)
    
    pipeline.fit(X_tr_temp, y_tr_temp)
    preds = pipeline.predict(X_te)
    
    results.append({
        "Model": name, 
        "Strategy": "Train-Val-Test (70-15-15)",
        "R2_Score": round(r2_score(y_te, preds), 4),
        "MAE": round(mean_absolute_error(y_te, preds), 2)
    })

# %%
results_df = pd.DataFrame(results)
print("\\n================ V4 ECONOMY SCOREBOARD ================")
print(results_df.to_string(index=False))

results_df.to_csv('../data/modeling_results_v4_economy.csv', index=False)
print("\\nModeling V4 complete. Scoreboard saved to data/modeling_results_v4_economy.csv")
