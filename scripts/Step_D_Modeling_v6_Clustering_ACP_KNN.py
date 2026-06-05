import pandas as pd
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline

print("Loading dataset for Global ACP + KNN Test...")
df = pd.read_csv('../data/moteur_ma_preprocessed_v5.csv')

X = df.drop('Price_MAD', axis=1)
y = df['Price_MAD']
cat_cols = ['Brand_Model', 'Transmission', 'Fuel', 'Location_Clean']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Test with 3 PCA Vectors and 4 PCA Vectors
for n_vectors in [3, 4]:
    print(f"\\n--- Testing with {n_vectors} PCA Vectors ---")
    
    pipeline = Pipeline([
        ('encoder', TargetEncoder(cols=cat_cols)),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_vectors)), 
        ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))
    ])
    
    pipeline.fit(X_tr, y_tr)
    preds = pipeline.predict(X_te)
    
    mae = mean_absolute_error(y_te, preds)
    mape = mean_absolute_percentage_error(y_te, preds) * 100
    
    print(f"Global MAE: {mae:,.2f} MAD")
    print(f"Global MAPE: {mape:.2f}%")

print("\\nDone!")
