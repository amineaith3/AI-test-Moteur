# Moroccan Second-Hand Car Price Prediction 🚗🇲🇦

This repository contains the full data science pipeline for estimating the fair price of second-hand cars in Morocco, strictly based on historical data scraped from [moteur.ma](https://www.moteur.ma).

This project fulfills the requirements of the academic assignment detailed in the [Project Requirements PDF](./Practical%20project_second_hand_car_price_SDBDIA2A.pdf).

## 📂 Project Structure

```text
car_price_project/
│
├── data/                                 # Contains raw and cleaned datasets
│   ├── moteur_ma_scraped_data.csv        # Massive raw dataset (~15,000 cars)
│   ├── moteur_ma_cleaned_500pages.csv    # Cleaned dataset without hidden prices (~13,180 cars)
│   └── moteur_ma_neuf_data.csv           # Reference dataset of new car baseline prices
│
├── images/                               # Exploratory Data Analysis (EDA) visualizations
│   ├── plot_1_top_brands.png
│   ├── plot_2_price_distribution.png
│   ├── plot_3_fuel_types.png
│   └── plot_4_avg_price_by_brand.png
│
├── scripts/                              # Core Python Logic
│   ├── PART_A_Scraper.py                 # Automates scraping 500+ pages of used cars
│   ├── PART_A_Scraper_Neuf.py            # Extracts exact prices for new car variants
│   ├── clean_data.py                     # Cleans corrupt rows (e.g., "Appeler pour le prix")
│   ├── PART_B_Exploratory_Analysis.py    # Jupyter-style notebook script for EDA
│   └── plot_insights.py                  # Generates matplotlib/seaborn image outputs
│
├── logs/                                 # Runtime logs for the scraping engines
│   └── scraper_output_500.log
│
├── requirements.txt                      # pip dependencies
└── Practical project_second_hand_car_price_SDBDIA2A.pdf  # Project Spec
```

## 🚀 Execution Roadmap

### Part A: Data Collection (Web Scraping) ✅
Built a custom BeautifulSoup scraper to paginate through thousands of car listings, successfully compiling a robust dataset of over 13,000 viable used cars. The raw data encompasses critical features like `Title`, `Price`, `Year`, `Mileage`, `Transmission`, `Fuel Type`, and `Location`.

### Part B: Exploratory Data Analysis (EDA) ✅
Performed deep data cleaning to parse strings into integers. Generated detailed visualizations detailing how car prices decay across mileage, identifying the dominant brands (Dacia, Renault, Peugeot), and highlighting the distribution of fuel types in the Moroccan market.

### Part C: Data Preprocessing ✅
Targeted numerical extraction (converting categorical variables into numerical weights via One-Hot Encoding) and handled extreme outliers. 

**Iterative Learning (V1 vs V2):** 
In our `v1` scripts, we only extracted the car `Brand` (e.g., Mercedes). This caused a massive Mean Absolute Error of **52,000 MAD** because the model couldn't distinguish between a 300k MAD A-Class and a 1.5M MAD G-Class! We documented this mistake, kept the V1 scripts for reference, and deployed `v2` scripts which explicitly extract and One-Hot encode the top 100 specific `CarModels` (e.g., Clio, Passat, G-Class).

### Part D & E: Predictive Modeling ✅
Implemented multiple Regression Models (Linear Regression, Ridge, Random Forest, Gradient Boosting). 
Tested across multiple strategies: **5-Fold CV**, **80-20**, **70-30**, **60-40**, and **70-15-15 splits**.

## 🛠️ Setup Instructions
1. Clone the repository.
2. Run `pip install -r requirements.txt`.
3. To visualize the data in an interactive window, execute `scripts/PART_B_Exploratory_Analysis.py` in VS Code using the `# %%` interactive cell feature.
