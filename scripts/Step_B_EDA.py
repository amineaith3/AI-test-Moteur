# %% [markdown]
# # Step B: Exploratory Data Analysis (EDA)
# In this notebook, we explore the raw 15,000+ car dataset. 
# We will visualize every major feature against the Price to understand which features drive the value of a car in Morocco.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create images folder if it doesn't exist
os.makedirs('../images', exist_ok=True)

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# %% [markdown]
# ## 1. Load Raw Dataset

# %%
df = pd.read_csv('../data/moteur_ma_scraped_data.csv')
print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

# %% [markdown]
# ## 2. Basic Cleaning for Visualization
# We have strings like "145,000 MAD" and "Appeler pour le prix". For EDA, we must convert these to numeric to plot them. 
# Full rigorous cleaning will happen in Step C.

# %%
# Drop hidden prices for EDA plots
df_eda = df[~df['Price'].str.contains('Appeler', case=False, na=False)].copy()

def clean_numeric(text):
    if pd.isna(text): return np.nan
    clean_str = ''.join(filter(str.isdigit, str(text)))
    return float(clean_str) if clean_str else np.nan

df_eda['Price_MAD'] = df_eda['Price'].apply(clean_numeric)
df_eda['Mileage_km'] = df_eda['Mileage'].apply(clean_numeric)
df_eda['Year'] = pd.to_numeric(df_eda['Year'], errors='coerce')

# Extract Brand
df_eda['Brand'] = df_eda['Title'].apply(lambda x: str(x).split()[0] if pd.notna(x) else 'Unknown')

# Filter extreme outliers just to make plots readable (keeping 98th percentile)
q98 = df_eda['Price_MAD'].quantile(0.98)
df_eda = df_eda[df_eda['Price_MAD'] <= q98]

# %% [markdown]
# ## 3. Feature vs. Price Visualizations

# %%
# 3.1 Price Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df_eda['Price_MAD'], bins=60, kde=True, color='#2ecc71')
plt.title('Distribution of Car Prices (MAD)')
plt.xlabel('Price (MAD)')
plt.ylabel('Frequency')
plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()
plt.savefig('../images/StepB_1_Price_Distribution.png')
plt.close()

# %%
# 3.2 Price vs Mileage
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df_eda, x='Mileage_km', y='Price_MAD', alpha=0.3, color='#e74c3c')
plt.title('Price vs. Mileage (Depreciation)')
plt.xlabel('Mileage (km)')
plt.ylabel('Price (MAD)')
plt.xlim(0, 400000) # Cap at 400k km for readability
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()
plt.savefig('../images/StepB_2_Price_vs_Mileage.png')
plt.close()

# %%
# 3.3 Price vs Year
plt.figure(figsize=(12, 5))
sns.boxplot(data=df_eda[df_eda['Year'] >= 2005], x='Year', y='Price_MAD', palette='viridis')
plt.title('Price vs. Year of Manufacture')
plt.xlabel('Year')
plt.ylabel('Price (MAD)')
plt.xticks(rotation=45)
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()
plt.savefig('../images/StepB_3_Price_vs_Year.png')
plt.close()

# %%
# 3.4 Price vs Transmission & Fuel
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=df_eda, x='Transmission', y='Price_MAD', ax=axes[0], palette='Set2')
axes[0].set_title('Average Price by Transmission')
axes[0].yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

sns.barplot(data=df_eda, x='Fuel', y='Price_MAD', ax=axes[1], palette='Set1')
axes[1].set_title('Average Price by Fuel Type')
axes[1].yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

plt.tight_layout()
plt.savefig('../images/StepB_4_Price_vs_Categoricals.png')
plt.close()

# %%
# 3.5 Price vs Brand
plt.figure(figsize=(14, 6))
top_brands = df_eda['Brand'].value_counts().head(15).index
sns.boxplot(data=df_eda[df_eda['Brand'].isin(top_brands)], x='Brand', y='Price_MAD', order=top_brands, palette='mako')
plt.title('Price Distribution across Top 15 Brands')
plt.xlabel('Brand')
plt.ylabel('Price (MAD)')
plt.xticks(rotation=45)
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()
plt.savefig('../images/StepB_5_Price_vs_Brand.png')
plt.close()

# %% [markdown]
# ### EDA Conclusions:
# 1. **Mileage:** Shows a strong negative correlation with Price.
# 2. **Year:** Shows an exponential positive correlation (newer cars retain significantly more value).
# 3. **Transmission:** Automatics command a massive premium over Manuals.
# 4. **Fuel:** Hybrids and Electrics are significantly more expensive than Diesel/Essence.
# 5. **Brand:** Luxury brands (Mercedes, BMW, Audi) have much higher baselines and wider variances than economy brands (Dacia, Renault).
# 
# *These features are all highly important. We will retain and encode all of them in Step C.*
