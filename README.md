# 🏎️ Moroccan Second-Hand Car Price Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sdbdia-group1.streamlit.app/)

> **Live Demo:** [https://sdbdia-group1.streamlit.app/](https://sdbdia-group1.streamlit.app/)

An end-to-end Data Science pipeline for predicting second-hand car prices in Morocco, trained on 15,000+ real listings scraped from `moteur.ma`. The final deployed model achieves a **13.2% MAPE** on everyday cars.

---

## 📂 Project Structure

```
car_price_project/
│
├── scripts/                        # All pipeline logic, chronologically versioned
│   ├── PART_A_Scraper.py           # Web scraper (BeautifulSoup + Requests)
│   ├── PART_A_Scraper_Neuf.py      # Updated scraper with improved pagination
│   ├── PART_B_Exploratory_Analysis.py  # EDA: statistics and distribution plots
│   ├── Step_B_EDA.py               # EDA: deeper correlation and feature plots
│   ├── Step_C_Preprocessing.py     # V1: Basic cleaning + One-Hot Encoding
│   ├── Step_C_Preprocessing_v2.py  # V2: Adds CarModel extraction
│   ├── Step_C_Preprocessing_v3.py  # V3: Target Encoding replaces boolean OHE
│   ├── Step_C_Preprocessing_v5.py  # V5: Brand+Model Feature Cross
│   ├── Step_D_Modeling.py          # V1: Baseline Random Forest
│   ├── Step_D_Modeling_v2.py       # V2: CarModel-aware model
│   ├── Step_D_Modeling_v3.py       # V3: Target Encoded model
│   ├── Step_D_Modeling_v4_Economy.py   # V4: Economy-only model
│   ├── Step_D_Modeling_v5_Hierarchical.py  # V5: 3-Stage Hierarchical Router (PRODUCTION)
│   ├── Step_D_Modeling_v6_Clustering_ACP_KNN.py  # V6: ACP+KNN experiment (benchmarked)
│   └── Step_E_Export_Model.py      # Exports the final V5 models to app/
│
├── data/                           # Raw and processed datasets
│   ├── moteur_ma_scraped_data.csv  # Raw 15,000+ listing dataset
│   └── moteur_ma_preprocessed_v5.csv   # Final V5 feature-engineered dataset
│
├── app/                            # Production Web Application
│   ├── app.py                      # Streamlit UI and routing logic
│   ├── model_classifier.pkl        # V5 Economy/Luxury Classifier (93% accuracy)
│   ├── model_economy.pkl           # V5 Economy Regressor (MAPE: 13.2%)
│   ├── model_luxury.pkl            # V5 Luxury Regressor (MAPE: 16.6%)
│   └── metadata.json               # Dynamic dropdown options for the UI
│
├── images/                         # EDA plots and charts
├── notebooks/                      # Jupytext-synced notebook copies
├── report.tex                      # Full academic LaTeX thesis (compile to PDF)
├── report.md                       # Markdown version of the report
└── requirements.txt                # Python dependencies
```

---

## 📖 The Full Story: From Scraping to Deployment

### Part A — Web Scraping
We built a custom `BeautifulSoup` scraper to extract car listings from `moteur.ma`. Initially we were blocked by pagination and server limits. We iterated on the scraper twice (`PART_A_Scraper.py` → `PART_A_Scraper_Neuf.py`) to handle retries and bypass rate limits. We capped at **500 pages** (15,000+ listings) — we could have gone to 1,200 pages, but we didn't want to burn anyone's laptop or risk an IP block.

**Extracted features:** `Title` (Brand + Model), `Price`, `Location`, `Year`, `Transmission`, `Fuel`, `Mileage`, `Link`

### Part B — Exploratory Data Analysis (EDA)
Two EDA scripts (`PART_B_Exploratory_Analysis.py` and `Step_B_EDA.py`) generate a full suite of descriptive plots saved to `images/`:
- **Price Distribution:** Most cars sit below 300k MAD, with a long luxury tail up to 1.5M MAD.
- **Depreciation Curve:** Prices drop steeply before 150,000 km, then flatten.
- **Categorical Premiums:** Automatics command a premium over manuals; Diesel dominates fuel types.

### Step C — Preprocessing Iterations
The preprocessing evolved across 4 major versions:

| Version | Key Change | MAE Result |
|---------|-----------|------------|
| V1 (`Step_C_Preprocessing.py`) | One-Hot Encoding, Brand only | 52,000 MAD |
| V2 (`Step_C_Preprocessing_v2.py`) | Extracted specific `CarModel` | 39,000 MAD |
| V3 (`Step_C_Preprocessing_v3.py`) | **Target Encoding** replaces boolean OHE | 35,000 MAD |
| V5 (`Step_C_Preprocessing_v5.py`) | **Feature Cross**: `Brand_Model` unified | Production |

> **Why Target Encoding?** One-Hot Encoding treated "Casablanca" and "El Jadida" as equal boolean switches. Target Encoding mathematically replaced each city with the average car price in that city, teaching the AI that northern urban centers are systematically more expensive.

### Step D — Modeling Iterations
All model evaluations used **5-Fold Cross Validation** in addition to standard train/test splits.

| Version | Architecture | Result |
|---------|-------------|--------|
| V1 | Random Forest, Brand only | R² = 0.58, MAE = 52k MAD |
| V2 | Random Forest + CarModel | R² = 0.70, MAE = 39k MAD |
| V3 | Random Forest + Target Encoding | R² = 0.76, MAE = 35k MAD |
| V4 | Economy-only model (≤300k MAD) | R² = 0.87, MAE = 16k MAD |
| **V5** | **3-Stage Hierarchical Router** | **MAPE: 13.2% (Eco) / 16.6% (Lux)** |
| V6 | K-Means + ACP (PCA) + KNN (benchmarked) | MAPE: ~24% ❌ |

#### V5: The 3-Stage Hierarchical Router (Production)
The final architecture uses a unified `Brand_Model` feature cross (e.g., "Mercedes-Benz Classe CLA" as a single entity) and routes predictions through:
1. **Classifier** (`model_classifier.pkl`) — Detects Economy vs. Luxury tier with **93% accuracy**
2. **Economy Regressor** (`model_economy.pkl`) — For cars ≤ 300k MAD, **MAPE: 13.2%**
3. **Luxury Regressor** (`model_luxury.pkl`) — For cars > 300k MAD, **MAPE: 16.6%**

#### V6: The ACP-KNN Experiment (Benchmarked, Not Deployed)
At the suggestion of an economics colleague, we tested whether **K-Means Clustering → PCA (ACP) → KNN** could outperform our Random Forest. We tried k=2, k=3 clusters and 3–4 PCA components. Result: the PCA "smeared" the pricing thresholds that KNN depends on, locking the global MAPE at **~24%** — nearly double our V5 error rate. This confirms that for tabular pricing data, tree-based ensembles (Random Forest) decisively outperform distance-based methods (KNN/PCA).

### Step E — Model Export
`Step_E_Export_Model.py` serializes the V5 pipelines using `joblib` and writes the dynamic dropdown metadata to `app/metadata.json`.

### Step F — Web App Deployment
The Streamlit app at `app/app.py` loads all three V5 `.pkl` files and routes predictions in real time. Hosted on **Streamlit Community Cloud**.

> 🔗 **Live App:** [https://sdbdia-group1.streamlit.app/](https://sdbdia-group1.streamlit.app/)

---

## 🛠️ How to Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/amineaith3/AI-test-Moteur.git
cd AI-test-Moteur
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Launch the Web App:**
```bash
cd app
streamlit run app.py
```

**4. Re-run the full pipeline from scratch:**
```bash
cd scripts
python PART_A_Scraper.py          # Scrape fresh data
python Step_B_EDA.py              # Generate EDA plots
python Step_C_Preprocessing_v5.py # Preprocess with V5 features
python Step_D_Modeling_v5_Hierarchical.py  # Train and export V5 models
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
category_encoders
joblib
streamlit
```

Install all with: `pip install -r requirements.txt`
