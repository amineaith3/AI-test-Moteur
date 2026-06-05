# %% [markdown]
# # Step C (Version 5): The Ultimate Preprocessing (Feature Crosses)
# We combine `Brand` and `CarModel` into a single, unbreakable `Brand_Model` feature.
# This explicitly prevents the AI from independently evaluating "Mercedes" and "Classe A",
# forcing it to evaluate the exact unified tier of "Mercedes-Benz Classe A".

# %%
import pandas as pd
import numpy as np

# %%
print("Loading raw data...")
df = pd.read_csv('../data/moteur_ma_scraped_data.csv')

df = df[~df['Price'].str.contains('Appeler', case=False, na=False)].copy()

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
# ## Feature Cross: Brand_Model

# %%
df['Brand'] = df['Title'].apply(lambda x: str(x).split()[0].title())
df['CarModel'] = df['Title'].apply(lambda x: " ".join(str(x).split()[1:]).title() if len(str(x).split()) > 1 else 'Unknown')

# Feature Cross!
df['Brand_Model'] = df['Brand'] + " " + df['CarModel']

# Keep top 250 Brand_Models to maintain high density but capture specific tiers
top_brand_models = df['Brand_Model'].value_counts().head(250).index
df['Brand_Model'] = df['Brand_Model'].apply(lambda x: x if x in top_brand_models else 'Other')

top_locations = df['Location'].value_counts().head(25).index
df['Location_Clean'] = df['Location'].apply(lambda x: x if x in top_locations else 'Other')

# We no longer need Brand and CarModel independently!
features = ['Brand_Model', 'Year', 'Mileage_km', 'Transmission', 'Fuel', 'Location_Clean']
target = 'Price_MAD'

df_model = df[features + [target]].copy()

output_path = '../data/moteur_ma_preprocessed_v5.csv'
df_model.to_csv(output_path, index=False)
print(f"\\nPreprocessing V5 complete! Data saved to {output_path}")
