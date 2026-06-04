import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re

# ==========================================
# PART A: Data Collection (New Cars / Neuf)
# ==========================================
# This script scrapes ALL new car versions from moteur.ma/fr/neuf/voiture/recherche
# This provides a clean dataset of 1200+ versions.

BASE_URL = "https://www.moteur.ma/fr/neuf/voiture/recherche"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def get_soup(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_price(text):
    """Finds price pattern like '224 900 Dhs' in a string."""
    match = re.search(r'([\d\s]+)\s*Dhs?', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(' ', '')
    return None

def scrape_neuf_cars(max_pages=36):
    all_versions = []
    
    for page in range(1, max_pages + 1):
        print(f"Scraping New Cars - Page {page}...")
        url = f"{BASE_URL}?page={page}"
        
        soup = get_soup(url)
        if not soup:
            continue
            
        cards = soup.find_all('div', class_='card')
        
        for card in cards:
            # Extract main car model name
            title_elem = card.find('h4')
            if not title_elem:
                continue
            
            main_title = title_elem.text.strip().replace('\n', '')
            # Clean warning messages if any
            if "Ce modèle n'est pas disponible" in main_title:
                main_title = main_title.split("Ce modèle")[0].strip()
            
            # Find the accordion list of versions
            version_items = card.find_all('li', class_='version-list-item')
            for v_item in version_items:
                version_data = {'Model': main_title}
                
                # Extract Version Name
                v_name_elem = v_item.find('div', class_='mb-1')
                if v_name_elem:
                    version_data['Version'] = v_name_elem.text.strip()
                    
                # Extract Badges (Fuel, Transmission, CV, CH)
                badges = v_item.find_all('span', class_='badge')
                for badge in badges:
                    text = badge.text.strip()
                    if 'CV' in text:
                        version_data['Fiscal_Power'] = text
                    elif 'CH' in text:
                        version_data['Horsepower'] = text
                    elif text in ['Essence', 'Diesel', 'Electrique', 'Hybride']:
                        version_data['Fuel'] = text
                    elif text in ['Automatique', 'Manuelle']:
                        version_data['Transmission'] = text
                        
                # Extract Price
                # The price is usually in the raw text of the item row (like "224 900 Dhs")
                price = extract_price(v_item.text)
                version_data['Price'] = price
                
                all_versions.append(version_data)
                
        # Be polite to the server
        time.sleep(random.uniform(0.5, 1.5))
        
    return pd.DataFrame(all_versions)

if __name__ == "__main__":
    print("Starting Web Scraper for NEW cars (Neuf)...")
    # Doing first 3 pages to test, change to 36 for full run
    df = scrape_neuf_cars(max_pages=3)
    
    if not df.empty:
        print(f"\\nSuccessfully scraped {len(df)} new car versions!")
        print(df.head())
        
        filename = "moteur_ma_neuf_data.csv"
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\\nData saved to {filename}")
    else:
        print("\\nNo data was scraped.")
