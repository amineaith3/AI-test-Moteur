# %% [markdown]
# # PART B: Exploratory Data Analysis (EDA)
# In this notebook, we will explore the second-hand car dataset scraped from `moteur.ma`.
# We will visualize the distributions, check for missing values, and understand the correlations between features and the target variable (Price).

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# %% [markdown]
# ## 1. Load the Dataset
# We load the data generated in Part A.

# %%
# Load data (ensure PART A scraper has finished running)
try:
    df = pd.read_csv("moteur_ma_scraped_data.csv")
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
except FileNotFoundError:
    print("Error: The CSV file was not found. Please wait for Part A to finish scraping.")
    df = pd.DataFrame() # Create empty DataFrame to prevent errors below

if not df.empty:
    display(df.head())

# %% [markdown]
# ## 2. Data Overview & Cleaning

# %%
if not df.empty:
    print("--- Data Info ---")
    df.info()
    
    print("\\n--- Missing Values ---")
    print(df.isnull().sum())

# %% [markdown]
# Notice that `Price`, `Mileage`, and `Year` are currently stored as strings containing characters like 'Dhs', 'km', and possibly missing values. We must clean these to perform numerical analysis.

# %%
def clean_price(price_str):
    if pd.isna(price_str):
        return np.nan
    # Remove 'Dhs', spaces, and anything non-numeric
    clean_str = ''.join(filter(str.isdigit, str(price_str)))
    return float(clean_str) if clean_str else np.nan

def clean_mileage(mileage_str):
    if pd.isna(mileage_str):
        return np.nan
    clean_str = ''.join(filter(str.isdigit, str(mileage_str)))
    return float(clean_str) if clean_str else np.nan

if not df.empty:
    df['Price_Clean'] = df['Price'].apply(clean_price)
    df['Mileage_Clean'] = df['Mileage'].apply(clean_mileage)
    
    # Convert Year to numeric
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    print("Cleaned Data Snapshot:")
    display(df[['Price', 'Price_Clean', 'Mileage', 'Mileage_Clean', 'Year']].head())

# %% [markdown]
# ## 3. Visualizations & Insights

# %%
# Distribution of Prices
if not df.empty:
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Price_Clean'].dropna(), bins=50, kde=True, color='blue')
    plt.title('Distribution of Car Prices (Dhs)')
    plt.xlabel('Price (Dhs)')
    plt.ylabel('Frequency')
    
    # Optional: Limiting x-axis to filter out extreme outliers for better visualization
    plt.xlim(0, df['Price_Clean'].quantile(0.95)) 
    plt.show()

# %%
# Mileage vs Price
if not df.empty:
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df, x='Mileage_Clean', y='Price_Clean', alpha=0.5, color='orange')
    plt.title('Impact of Mileage on Price')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price (Dhs)')
    
    # Limiting axes to remove extreme outliers
    plt.xlim(0, df['Mileage_Clean'].quantile(0.98))
    plt.ylim(0, df['Price_Clean'].quantile(0.98))
    plt.show()

# %%
# Average Price by Fuel Type
if not df.empty:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='Fuel', y='Price_Clean', ci=None, palette='viridis')
    plt.title('Average Price by Fuel Type')
    plt.xlabel('Fuel Type')
    plt.ylabel('Average Price (Dhs)')
    plt.show()

# %% [markdown]
# ### Insights Summary:
# - **Price Distribution:** Car prices are heavily right-skewed, meaning most cars are in the lower-to-middle price range, with a few very expensive luxury cars.
# - **Mileage vs. Price:** There is a clear negative correlation (depreciation). As mileage increases, the price tends to decrease.
# - **Fuel Type:** We can observe the premium/discount applied to Diesel vs. Essence or Hybrid vehicles.
# 
# *Next Step: We will handle missing values, cap outliers, encode categories, and prepare the dataset for predictive modeling in Part C.*
