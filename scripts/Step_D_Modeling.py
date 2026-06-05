# %% [markdown]
# # Step D & E: Modeling and Evaluation
# Here we will train 6 different Machine Learning models to predict car prices.
# We will evaluate them using multiple splitting strategies to understand how the amount of training data affects performance:
# 1. 5-Fold Cross Validation
# 2. 80-20 Train-Test Split
# 3. 70-30 Train-Test Split
# 4. 60-40 Train-Test Split
# 5. Train-Validation-Test (70-15-15) Split

# %%
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# %% [markdown]
# ## 1. Load Preprocessed Data

# %%
print("Loading preprocessed dataset...")
df = pd.read_csv('../data/moteur_ma_preprocessed.csv')

X = df.drop('Price_MAD', axis=1)
y = df['Price_MAD']

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# %% [markdown]
# ## 2. Define Models and Pipelines
# We use pipelines to ensure `StandardScaler` is applied properly after splitting (preventing data leakage).

# %%
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(max_iter=2000),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42), # n_estimators=50 for speed
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Create pipelines
pipelines = {}
for name, model in models.items():
    pipelines[name] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

# %% [markdown]
# ## 3. Experiment Engine: Evaluating Splits & Cross Validation

# %%
results = []

def evaluate_split(model_name, pipeline, X_train, X_test, y_train, y_test, split_name):
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    return {
        "Model": model_name,
        "Strategy": split_name,
        "R2_Score": round(r2, 4),
        "MAE": round(mae, 2),
        "Time_sec": round(time.time() - start_time, 2)
    }

print("\\nStarting Modeling Experiments... This may take a few minutes!\\n")

for name, pipeline in pipelines.items():
    print(f"Training {name}...")
    
    # --- 1. 5-Fold Cross Validation ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # cross_val_score uses negative MAE by default
    cv_r2 = cross_val_score(pipeline, X, y, cv=kf, scoring='r2').mean()
    cv_mae = -cross_val_score(pipeline, X, y, cv=kf, scoring='neg_mean_absolute_error').mean()
    
    results.append({
        "Model": name,
        "Strategy": "5-Fold CV",
        "R2_Score": round(cv_r2, 4),
        "MAE": round(cv_mae, 2),
        "Time_sec": "N/A" # CV handles time internally
    })
    
    # --- 2. 80/20 Split ---
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
    results.append(evaluate_split(name, pipeline, X_tr, X_te, y_tr, y_te, "80-20 Split"))
    
    # --- 3. 70/30 Split ---
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=42)
    results.append(evaluate_split(name, pipeline, X_tr, X_te, y_tr, y_te, "70-30 Split"))
    
    # --- 4. 60/40 Split ---
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.40, random_state=42)
    results.append(evaluate_split(name, pipeline, X_tr, X_te, y_tr, y_te, "60-40 Split"))
    
    # --- 5. Train-Validation-Test (70-15-15) ---
    # First split off 30% for Val+Test
    X_tr_temp, X_val_test, y_tr_temp, y_val_test = train_test_split(X, y, test_size=0.30, random_state=42)
    # Split the 30% into 15% Val and 15% Test
    X_val, X_te, y_val, y_te = train_test_split(X_val_test, y_val_test, test_size=0.50, random_state=42)
    
    # We train on Train, validate on Val (simulated here just by fitting on train), test on Test
    # In deep learning, validation is used to stop training early. Here we just measure it.
    start_time = time.time()
    pipeline.fit(X_tr_temp, y_tr_temp) # Train on 70%
    val_preds = pipeline.predict(X_val) # Validate on 15%
    test_preds = pipeline.predict(X_te) # Final Test on 15%
    
    test_r2 = r2_score(y_te, test_preds)
    test_mae = mean_absolute_error(y_te, test_preds)
    
    results.append({
        "Model": name,
        "Strategy": "Train-Val-Test (70-15-15)",
        "R2_Score": round(test_r2, 4),
        "MAE": round(test_mae, 2),
        "Time_sec": round(time.time() - start_time, 2)
    })

# %% [markdown]
# ## 4. Display Final Scoreboard

# %%
results_df = pd.DataFrame(results)
print("\\n================ EXPERIMENT SCOREBOARD ================")
print(results_df.to_string(index=False))

# Save results for Step F (Business Insights)
results_df.to_csv('../data/modeling_results.csv', index=False)
print("\\n✅ Modeling complete. Scoreboard saved to data/modeling_results.csv")
