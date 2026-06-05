# %% [markdown]
# # Step C: Data Preprocessing
# In this notebook, we prepare the dataset for Machine Learning.
# Steps include:
# 1. Dropping missing or hidden values.
# 2. Extracting numerical values from strings.
# 3. Handling Outliers.
# 4. Feature Engineering & Categorical Encoding.

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## 1. Data Cleaning
# We load the raw 15,000+ car dataset and remove rows that don't have a valid price.

# %%
print("Loading raw data...")
df = pd.read_csv('../data/moteur_ma_scraped_data.csv')

# Drop rows where Price is hidden
df = df[~df['Price'].str.contains('Appeler', case=False, na=False)].copy()

# Function to extract numeric values from strings like "145,000 MAD"
def extract_numeric(text):
    if pd.isna(text): return np.nan
    clean_str = ''.join(filter(str.isdigit, str(text)))
    return float(clean_str) if clean_str else np.nan

df['Price_MAD'] = df['Price'].apply(extract_numeric)
df['Mileage_km'] = df['Mileage'].apply(extract_numeric)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Drop missing values in critical columns
df.dropna(subset=['Price_MAD', 'Mileage_km', 'Year', 'Transmission', 'Fuel'], inplace=True)

# %% [markdown]
# ## 2. Outlier Removal
# We remove obvious errors (e.g., cars listed for 10 MAD or 10,000,000 MAD).

# %%
# Keep realistic prices: 20,000 MAD to 1,500,000 MAD
df = df[(df['Price_MAD'] >= 20000) & (df['Price_MAD'] <= 1500000)]

# Keep realistic mileage: less than 500,000 km
df = df[df['Mileage_km'] <= 500000]

# Keep realistic years: 1990 to Current
df = df[(df['Year'] >= 1990) & (df['Year'] <= 2026)]

print(f"Dataset size after cleaning and outlier removal: {len(df)} cars")

# %% [markdown]
# ## 3. Feature Engineering & Categorical Encoding
# Machine Learning models need numbers, not strings. We will One-Hot Encode categories.

# %%
# Extract Brand from Title
df['Brand'] = df['Title'].apply(lambda x: str(x).split()[0].title())

# There are many rare brands. We'll keep the top 20 and label the rest as 'Other'
top_brands = df['Brand'].value_counts().head(20).index
df['Brand'] = df['Brand'].apply(lambda x: x if x in top_brands else 'Other')

# We can also keep top locations and label the rest 'Other'
top_locations = df['Location'].value_counts().head(10).index
df['Location_Clean'] = df['Location'].apply(lambda x: x if x in top_locations else 'Other')

# Select the features we want to use for modeling
features = ['Brand', 'Year', 'Mileage_km', 'Transmission', 'Fuel', 'Location_Clean']
target = 'Price_MAD'

# Create a clean DataFrame for modeling
df_model = df[features + [target]].copy()

# One-Hot Encode categorical variables
# drop_first=True helps avoid the dummy variable trap (multicollinearity)
df_encoded = pd.get_dummies(df_model, columns=['Brand', 'Transmission', 'Fuel', 'Location_Clean'], drop_first=True)

# %% [markdown]
# ## 4. Final Verification and Export

# %%
print("\\n--- Final Preprocessed Dataset ---")
print(f"Shape: {df_encoded.shape}")
print(df_encoded.head())

# Save to CSV for Step D
output_path = '../data/moteur_ma_preprocessed.csv'
df_encoded.to_csv(output_path, index=False)
print(f"\\n✅ Preprocessing complete! Data saved to {output_path}")
