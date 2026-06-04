import pandas as pd

# Load the massive 500-page dataset
df = pd.read_csv('moteur_ma_scraped_data.csv')
total_rows = len(df)

# Drop rows where the price is "Appeler pour le prix" or similar non-numeric indicators
df_cleaned = df[~df['Price'].str.contains('Appeler', case=False, na=False)]

# Save the cleaned dataset
df_cleaned.to_csv('moteur_ma_cleaned_500pages.csv', index=False, encoding='utf-8')

dropped_rows = total_rows - len(df_cleaned)
print(f"Total cars originally scraped: {total_rows}")
print(f"Cars with 'Appeler pour le prix' dropped: {dropped_rows}")
print(f"Total usable cars left: {len(df_cleaned)}")
