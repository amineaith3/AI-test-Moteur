# %% [markdown]
# # Step C (Version 3): Data Preprocessing without Booleans
# The True/False (One-Hot Encoding) approach fractures the dataset into 138 columns and removes ordinal context 
# (e.g. treating Casa and El Jadida as just random True/False switches instead of acknowledging regional price differences).
# In V3, we will NOT use One-Hot Encoding. We will just clean the text categories and export them as raw strings.
# The mathematical weighting of these categories will happen dynamically in Step D using Target Encoding.

# %%
import pandas as pd
import numpy as np

# %%
print("Loading raw data...")
df = pd.read_csv('../data/moteur_ma_scraped_data.csv')

# Drop hidden prices
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
# ## Feature Engineering: Raw String Extraction
# We extract the categories but leave them as words.

# %%
df['Brand'] = df['Title'].apply(lambda x: str(x).split()[0].title())
df['CarModel'] = df['Title'].apply(lambda x: " ".join(str(x).split()[1:]).title() if len(str(x).split()) > 1 else 'Unknown')

top_brands = df['Brand'].value_counts().head(30).index
df['Brand'] = df['Brand'].apply(lambda x: x if x in top_brands else 'Other')

# We can keep more models now because Target Encoding handles high-cardinality perfectly
top_models = df['CarModel'].value_counts().head(200).index
df['CarModel'] = df['CarModel'].apply(lambda x: x if x in top_models else 'Other')

# Keep top 45 Locations to capture the North/South price gaps
top_locations = df['Location'].value_counts().head(45).index
df['Location_Clean'] = df['Location'].apply(lambda x: x if x in top_locations else 'Other')

features = ['Brand', 'CarModel', 'Year', 'Mileage_km', 'Transmission', 'Fuel', 'Location_Clean']
target = 'Price_MAD'

df_model = df[features + [target]].copy()

# Look! No pd.get_dummies! We stay at 7 beautiful columns.
print(f"Shape after V3 Cleaning: {df_model.shape}")

output_path = '../data/moteur_ma_preprocessed_v3.csv'
df_model.to_csv(output_path, index=False)
print(f"\\nPreprocessing V3 complete! Data saved to {output_path}")
