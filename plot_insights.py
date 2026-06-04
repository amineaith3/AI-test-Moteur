import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("Loading the dataset...")
df = pd.read_csv('moteur_ma_cleaned_500pages.csv')

print("Cleaning data for plotting...")
# Clean Price
def clean_price(price_str):
    if pd.isna(price_str): return np.nan
    clean_str = ''.join(filter(str.isdigit, str(price_str)))
    return float(clean_str) if clean_str else np.nan

df['Price_Clean'] = df['Price'].apply(clean_price)

# Extract Brand (first word of the title)
df['Brand'] = df['Title'].apply(lambda x: str(x).split()[0] if pd.notna(x) else 'Unknown')

# Filter out extreme outliers in price for better visualization (e.g. keeping 98th percentile)
q98 = df['Price_Clean'].quantile(0.98)
df_filtered = df[df['Price_Clean'] <= q98]

print("Generating Plot 1: Car Brands Distribution...")
plt.figure(figsize=(12, 6))
top_brands = df['Brand'].value_counts().head(15)
sns.barplot(x=top_brands.values, y=top_brands.index, palette='viridis')
plt.title('Top 15 Most Listed Car Brands')
plt.xlabel('Number of Listings')
plt.ylabel('Brand')
plt.tight_layout()
plt.savefig('plot_1_top_brands.png')
plt.close()

print("Generating Plot 2: Price Interval Distribution...")
plt.figure(figsize=(12, 6))
sns.histplot(df_filtered['Price_Clean'], bins=50, kde=True, color='blue')
plt.title('Distribution of Car Prices (Excluding Extreme Outliers)')
plt.xlabel('Price (Dhs)')
plt.ylabel('Frequency')
# Format x-axis with commas for readability
plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()
plt.savefig('plot_2_price_distribution.png')
plt.close()

print("Generating Plot 3: Fuel Type Distribution...")
plt.figure(figsize=(8, 6))
fuel_counts = df['Fuel'].value_counts()
sns.barplot(x=fuel_counts.index, y=fuel_counts.values, palette='magma')
plt.title('Listings by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Number of Listings')
plt.tight_layout()
plt.savefig('plot_3_fuel_types.png')
plt.close()

print("Generating Plot 4: Average Price by Top Brands...")
plt.figure(figsize=(14, 7))
# Only take the top 15 brands by frequency
top_15_brands = df['Brand'].value_counts().head(15).index
df_top_brands = df_filtered[df_filtered['Brand'].isin(top_15_brands)]

order = df_top_brands.groupby('Brand')['Price_Clean'].mean().sort_values(ascending=False).index
sns.barplot(data=df_top_brands, x='Brand', y='Price_Clean', order=order, palette='crest', errorbar=None)
plt.title('Average Price for Top 15 Brands')
plt.xlabel('Brand')
plt.ylabel('Average Price (Dhs)')
plt.xticks(rotation=45)
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()
plt.savefig('plot_4_avg_price_by_brand.png')
plt.close()

print("All plots generated and saved as PNG images successfully!")
