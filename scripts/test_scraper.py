import requests
from bs4 import BeautifulSoup
import json

url = "https://www.moteur.ma/fr/voiture/achat-voiture-occasion/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

links = soup.find_all('a', href=True)
unique_links = list(set([l['href'] for l in links if 'moteur.ma' in l['href'] or l['href'].startswith('/')]))
print("Found links:")
for l in unique_links[:20]:
    print(l)


