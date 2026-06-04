import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

# ==========================================
# PART A: Data Collection (Web Scraping)
# ==========================================
# This script scrapes second-hand car listings from moteur.ma
# Features to collect: Brand/Model, Year, Mileage, Fuel type, Transmission, Engine power, Location, Condition, Price

BASE_URL = "https://www.moteur.ma/fr/voiture/achat-voiture-occasion/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_soup(url):
    """Fetches the URL and returns a BeautifulSoup object."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_car_details(car_card):
    """Extracts features from a single car listing card."""
    car_data = {}
    
    try:
        # Title usually contains Brand and Model
        title_elem = car_card.find('h5', class_='ads-index-title')
        if title_elem:
            car_data['Title'] = title_elem.text.strip()
        else:
            return None # Invalid card
        
        # Price is usually highlighted
        price_elem = car_card.find('h4', class_='ad-price-grid')
        car_data['Price'] = price_elem.text.strip().replace('\\n', ' ').replace('\\r', '') if price_elem else None
        
        # Location
        desc_div = car_card.find('div', class_='item-card9-desc')
        if desc_div:
            loc_a = desc_div.find('a')
            car_data['Location'] = loc_a.text.strip() if loc_a else None
            
        # Meta info contains Year, Transmission, Fuel, Mileage
        meta_div = car_card.find('div', class_='ad-meta')
        if meta_div:
            spans = meta_div.find_all('span')
            for span in spans:
                text = span.text.strip()
                icon = span.find('i')
                if icon:
                    classes = icon.get('class', [])
                    if 'fa-calendar' in classes:
                        car_data['Year'] = text
                    elif 'fa-cog' in classes:
                        car_data['Transmission'] = text
                    elif 'fa-tachometer' in classes:
                        car_data['Fuel'] = text
                    elif 'fa-road' in classes:
                        car_data['Mileage'] = text
                        
    except Exception as e:
        print(f"Error extracting card details: {e}")
        
    return car_data

def scrape_moteur_ma(max_pages=5):
    """Scrapes multiple pages and returns a Pandas DataFrame."""
    all_cars = []
    
    for page in range(1, max_pages + 1):
        print(f"Scraping page {page}...")
        url = f"{BASE_URL}?page={page}" if page > 1 else BASE_URL
        
        soup = get_soup(url)
        if not soup:
            continue
            
        cards = soup.find_all('div', class_='ads-index-card')
        for card in cards:
            link_elem = card.find('a', href=lambda x: x and 'detail-annonce' in x)
            link = link_elem['href'] if link_elem else None
            
            car_info = extract_car_details(card)
            if car_info and car_info.get('Title'):
                car_info['Link'] = link
                all_cars.append(car_info)
        
        # Respectful delay between requests
        time.sleep(random.uniform(1, 3))
        
    return pd.DataFrame(all_cars)

if __name__ == "__main__":
    print("Starting Web Scraper for moteur.ma...")
    # Scrape 500 pages for a massive dataset
    df = scrape_moteur_ma(max_pages=500)
    
    if not df.empty:
        print(f"\\nSuccessfully scraped {len(df)} cars!")
        print(df.head())
        
        # Save to CSV for Part B
        filename = "../data/moteur_ma_scraped_data.csv"
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\\nData saved to {filename}")
    else:
        print("\\nNo data was scraped. CSS selectors might need updating.")
