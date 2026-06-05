# %% [markdown]
# # Step D (Version 2): Modeling with Car Models
# Evaluating models on the V2 dataset which includes the heavily requested `CarModel` feature.
# We expect the 52k MAD error to drop significantly.

# %%
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# %%
print("Loading V2 preprocessed dataset...")
df = pd.read_csv('../data/moteur_ma_preprocessed_v2.csv')

X = df.drop('Price_MAD', axis=1)
y = df['Price_MAD']

# We limit to the top performing models from V1 to save time
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=42)
}

pipelines = {name: Pipeline([('scaler', StandardScaler()), ('model', model)]) for name, model in models.items()}

# %%
results = []
def evaluate_split(model_name, pipeline, X_train, X_test, y_train, y_test, split_name):
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    return {
        "Model": model_name,
        "Strategy": split_name,
        "R2_Score": round(r2_score(y_test, preds), 4),
        "MAE": round(mean_absolute_error(y_test, preds), 2)
    }

print("Training V2 Models...")

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
print("\\n================ V2 EXPERIMENT SCOREBOARD ================")
print(results_df.to_string(index=False))

results_df.to_csv('../data/modeling_results_v2.csv', index=False)
print("\\nModeling V2 complete. Scoreboard saved to data/modeling_results_v2.csv")
