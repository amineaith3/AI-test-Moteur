import pandas as pd
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

df = pd.read_csv('data/moteur_ma_preprocessed_v5.csv')
cat_cols = ['Brand_Model', 'Transmission', 'Fuel', 'Location_Clean']

# Economy
df_economy = df[df['Price_MAD'] <= 300000]
X_eco = df_economy.drop('Price_MAD', axis=1)
y_eco = df_economy['Price_MAD']

eco_pipeline = Pipeline([
    ('encoder', TargetEncoder(cols=cat_cols)),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42))
])
X_e_tr, X_e_te, y_e_tr, y_e_te = train_test_split(X_eco, y_eco, test_size=0.2, random_state=42)
eco_pipeline.fit(X_e_tr, y_e_tr)
eco_preds = eco_pipeline.predict(X_e_te)
eco_mape = mean_absolute_percentage_error(y_e_te, eco_preds) * 100

# Luxury
df_luxury = df[df['Price_MAD'] > 300000]
X_lux = df_luxury.drop('Price_MAD', axis=1)
y_lux = df_luxury['Price_MAD']

lux_pipeline = Pipeline([
    ('encoder', TargetEncoder(cols=cat_cols)),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42))
])
X_l_tr, X_l_te, y_l_tr, y_l_te = train_test_split(X_lux, y_lux, test_size=0.2, random_state=42)
lux_pipeline.fit(X_l_tr, y_l_tr)
lux_preds = lux_pipeline.predict(X_l_te)
lux_mape = mean_absolute_percentage_error(y_l_te, lux_preds) * 100

print(f"Economy MAPE: {eco_mape:.2f}%")
print(f"Luxury MAPE: {lux_mape:.2f}%")
