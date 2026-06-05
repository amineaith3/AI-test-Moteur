# Moroccan Second-Hand Car Price Predictor: Final Project Report

**Author:** Amine
**Project Focus:** Predicting second-hand car prices specifically for the Moroccan market using AI models trained on over 15,000 live listings scraped from moteur.ma.

---

## 1. Executive Summary
This project outlines the end-to-end Data Science pipeline used to accurately estimate the fair market value of used cars in Morocco. By transitioning from basic data scraping to complex mathematical modeling via Random Forests and Target Encoding, we successfully created an AI model that predicts car prices with **87.3% accuracy**, within a highly competitive margin of error of just **±16,390 MAD**.

## 2. Iterative Pipeline Evolution

### Step A: Data Collection
A Python web scraper (`BeautifulSoup` + `Requests`) was designed to parse the HTML structure of `moteur.ma`. We bypassed generic limitations to scrape over 500 pages, securing a raw dataset of over 15,000 real listings.

### Step B: Exploratory Data Analysis (EDA)
Using `matplotlib` and `seaborn`, we mapped features against price. Key findings:
- **Depreciation Curve:** Prices heavily drop before 150,000 km and flatten afterward.
- **Premium Features:** Automatic transmission and Hybrid fuel demand heavy premiums.

### Step C & D: Preprocessing and Modeling Iterations
The project underwent intense iterative learning to fix real-world problems:
- **Version 1:** Initial model. Errored heavily (MAE: 52,000 MAD) because the algorithm only knew the car's Brand, not the Model.
- **Version 2:** Extracted the specific `CarModel` (e.g., "Clio" instead of just "Renault"). Accuracy shot up to **70%**, bringing error down to ~39,000 MAD.
- **Version 3:** Replaced boolean true/false values with **Target Encoding**, teaching the AI that cities like Casablanca mathematically carry higher baseline prices than El Jadida. Accuracy hit **76.6%** (MAE: 35k MAD).
- **Version 4 (Final):** Discovered that huge errors on 1.5 million MAD luxury cars were ruining the average accuracy of cheap cars. We isolated the dataset into an **Economy subset** (< 300,000 MAD). The final Random Forest model achieved a massive **87.3% accuracy** and a remarkable **16,390 MAD error margin**.

## 3. Evaluation & Validation
We rigorously tested the models across:
1. 80/20, 70/30, and 60/40 splits.
2. 70/15/15 Train-Validation-Test splits.
3. **5-Fold Cross Validation (The chosen metric)** to ensure the model wasn't just getting lucky on an easy test set.

## 4. Business Insights & Web Application
The final pipeline was serialized using `joblib` and deployed into a **Streamlit Web Application**. The app allows end-users to select their exact car features and receive an instant, mathematically sound price prediction. 

The tool proves that with robust data scaling and regional targeting, AI can successfully regulate and evaluate the chaotic Moroccan second-hand car market.
