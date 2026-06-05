# %% [markdown]
# # Step C (Version 2): Data Preprocessing with Car Models
# In our first iteration, we only extracted the 'Brand'. This led to an unacceptable mean error of 52,000 MAD 
# because the model couldn't distinguish between a Mercedes A-Class and a Mercedes G-Class!
# Here, we fix this by extracting the specific 'Car Model'.

# %%
import pandas as pd
import numpy as np

# %%
print("Loading raw data...")
df = pd.read_csv('../data/moteur_ma_scraped_data.csv')

# Drop rows where Price is hidden
df = df[~df['Price'].str.contains('Appeler', case=False, na=False)].copy()

# Extract numeric values
def extract_numeric(text):
    if pd.isna(text): return np.nan
    clean_str = ''.join(filter(str.isdigit, str(text)))
    return float(clean_str) if clean_str else np.nan

df['Price_MAD'] = df['Price'].apply(extract_numeric)
df['Mileage_km'] = df['Mileage'].apply(extract_numeric)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

df.dropna(subset=['Price_MAD', 'Mileage_km', 'Year', 'Transmission', 'Fuel'], inplace=True)

# Outlier Removal
df = df[(df['Price_MAD'] >= 20000) & (df['Price_MAD'] <= 1500000)]
df = df[df['Mileage_km'] <= 500000]
df = df[(df['Year'] >= 1990) & (df['Year'] <= 2025)]

# %% [markdown]
# ## Feature Engineering: Brand AND Model
# We now extract everything after the first word as the specific car model.

# %%
# Extract Brand
df['Brand'] = df['Title'].apply(lambda x: str(x).split()[0].title())

# Extract Car Model (everything after the Brand)
df['CarModel'] = df['Title'].apply(lambda x: " ".join(str(x).split()[1:]).title() if len(str(x).split()) > 1 else 'Unknown')

# Keep top 20 Brands
top_brands = df['Brand'].value_counts().head(20).index
df['Brand'] = df['Brand'].apply(lambda x: x if x in top_brands else 'Other')

# Keep top 100 Car Models (crucial for accuracy)
top_models = df['CarModel'].value_counts().head(100).index
df['CarModel'] = df['CarModel'].apply(lambda x: x if x in top_models else 'Other')

# Keep top 10 Locations
top_locations = df['Location'].value_counts().head(10).index
df['Location_Clean'] = df['Location'].apply(lambda x: x if x in top_locations else 'Other')

# Select features
features = ['Brand', 'CarModel', 'Year', 'Mileage_km', 'Transmission', 'Fuel', 'Location_Clean']
target = 'Price_MAD'

df_model = df[features + [target]].copy()

# One-Hot Encode
df_encoded = pd.get_dummies(df_model, columns=['Brand', 'CarModel', 'Transmission', 'Fuel', 'Location_Clean'], drop_first=True)

print(f"Shape after V2 Encoding: {df_encoded.shape}")

output_path = '../data/moteur_ma_preprocessed_v2.csv'
df_encoded.to_csv(output_path, index=False)
print(f"\\nPreprocessing V2 complete! Data saved to {output_path}")
