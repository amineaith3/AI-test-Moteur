"""
Generate all report images:
  1. 6 x Feature vs Price scatter/box plots
  2. Split strategy comparison table (V1 data)
  3. Algorithm comparison table (V1 preprocessing)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Styling ---
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.facecolor': '#1e1e2e',
    'figure.facecolor': '#13131f',
    'axes.edgecolor': '#444466',
    'axes.labelcolor': 'white',
    'xtick.color': '#aaaacc',
    'ytick.color': '#aaaacc',
    'text.color': 'white',
    'grid.color': '#2a2a3e',
    'grid.linewidth': 0.6,
})
ACCENT = '#7c6af7'
ACCENT2 = '#f97c6a'

# Load Data
print("Loading data...")
df = pd.read_csv('../data/moteur_ma_preprocessed_v5.csv')
df_raw = df.copy()

# =========================================================
# 1. Feature vs Price Plots (6 plots)
# =========================================================
print("Generating Feature vs Price plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Feature vs Price Analysis', fontsize=20, fontweight='bold', color='white', y=1.01)

# Plot 1: Price vs Year
ax = axes[0, 0]
year_avg = df.groupby('Year')['Price_MAD'].median().reset_index()
ax.scatter(df['Year'], df['Price_MAD'] / 1000, alpha=0.15, color=ACCENT, s=8)
ax.plot(year_avg['Year'], year_avg['Price_MAD'] / 1000, color=ACCENT2, linewidth=2, label='Median')
ax.set_title('Price vs Manufacture Year', fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Price (k MAD)')
ax.legend()
ax.grid(True)

# Plot 2: Price vs Mileage
ax = axes[0, 1]
ax.scatter(df['Mileage_km'] / 1000, df['Price_MAD'] / 1000, alpha=0.1, color=ACCENT, s=8)
ax.set_title('Price vs Mileage (Depreciation Curve)', fontweight='bold')
ax.set_xlabel('Mileage (×1000 km)')
ax.set_ylabel('Price (k MAD)')
ax.grid(True)

# Plot 3: Price vs Fuel Type
ax = axes[0, 2]
fuel_order = df.groupby('Fuel')['Price_MAD'].median().sort_values(ascending=False).index
colors = [ACCENT, ACCENT2, '#6af7c8', '#f7c86a', '#f76a8e']
for i, fuel in enumerate(fuel_order):
    data = df[df['Fuel'] == fuel]['Price_MAD'] / 1000
    ax.boxplot(data, positions=[i], widths=0.6,
               patch_artist=True,
               boxprops=dict(facecolor=colors[i % len(colors)], alpha=0.7),
               medianprops=dict(color='white', linewidth=2),
               whiskerprops=dict(color='#aaaacc'),
               capprops=dict(color='#aaaacc'),
               flierprops=dict(marker='.', color='#aaaacc', alpha=0.3))
ax.set_xticks(range(len(fuel_order)))
ax.set_xticklabels(fuel_order, rotation=20, ha='right')
ax.set_title('Price Distribution by Fuel Type', fontweight='bold')
ax.set_ylabel('Price (k MAD)')
ax.grid(True, axis='y')

# Plot 4: Price vs Transmission
ax = axes[1, 0]
trans_order = df.groupby('Transmission')['Price_MAD'].median().sort_values(ascending=False).index
for i, t in enumerate(trans_order):
    data = df[df['Transmission'] == t]['Price_MAD'] / 1000
    ax.boxplot(data, positions=[i], widths=0.5,
               patch_artist=True,
               boxprops=dict(facecolor=colors[i % len(colors)], alpha=0.7),
               medianprops=dict(color='white', linewidth=2),
               whiskerprops=dict(color='#aaaacc'),
               capprops=dict(color='#aaaacc'),
               flierprops=dict(marker='.', color='#aaaacc', alpha=0.3))
ax.set_xticks(range(len(trans_order)))
ax.set_xticklabels(trans_order, rotation=15)
ax.set_title('Price Distribution by Transmission', fontweight='bold')
ax.set_ylabel('Price (k MAD)')
ax.grid(True, axis='y')

# Plot 5: Price vs Location (Top 10)
ax = axes[1, 1]
top_locs = df.groupby('Location_Clean')['Price_MAD'].median().sort_values(ascending=False).head(10)
bars = ax.barh(top_locs.index[::-1], top_locs.values[::-1] / 1000, color=ACCENT, alpha=0.85)
for bar, val in zip(bars, top_locs.values[::-1] / 1000):
    ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
            f'{val:.0f}k', va='center', fontsize=8, color='white')
ax.set_title('Median Price by Location (Top 10)', fontweight='bold')
ax.set_xlabel('Median Price (k MAD)')
ax.grid(True, axis='x')

# Plot 6: Price vs Top 10 Brands
ax = axes[1, 2]
df['Brand'] = df['Brand_Model'].apply(lambda x: x.split()[0] if len(x.split()) > 0 else x)
top_brands = df.groupby('Brand')['Price_MAD'].median().sort_values(ascending=False).head(10)
bars = ax.barh(top_brands.index[::-1], top_brands.values[::-1] / 1000, color=ACCENT2, alpha=0.85)
for bar, val in zip(bars, top_brands.values[::-1] / 1000):
    ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
            f'{val:.0f}k', va='center', fontsize=8, color='white')
ax.set_title('Median Price by Brand (Top 10)', fontweight='bold')
ax.set_xlabel('Median Price (k MAD)')
ax.grid(True, axis='x')

plt.tight_layout()
plt.savefig('../images/report_feature_vs_price.png', dpi=150, bbox_inches='tight',
            facecolor='#13131f')
print("  -> Saved: images/report_feature_vs_price.png")
plt.close()

# =========================================================
# 2. Split Strategy Comparison (V1 baseline)
# =========================================================
print("Running split strategy comparison on V1 data...")

cat_cols = ['Brand_Model', 'Transmission', 'Fuel', 'Location_Clean']
X = df.drop(['Price_MAD', 'Brand'], axis=1, errors='ignore')
y = df['Price_MAD']

results = {}

# Simple splits
for name, (test_size, val_size) in [
    ('80-20', (0.20, None)),
    ('70-30', (0.30, None)),
    ('60-40', (0.40, None)),
]:
    pipe = Pipeline([
        ('enc', TargetEncoder(cols=cat_cols)),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42))
    ])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    results[name] = {
        'MAE': mean_absolute_error(y_te, preds),
        'R2': r2_score(y_te, preds),
        'Test Size': f'{int(test_size*100)}%'
    }

# 70-20-10 and 70-15-15
for name, (val_size, test_size) in [
    ('70-20-10', (0.20, 0.10)),
    ('70-15-15', (0.15, 0.15)),
]:
    pipe = Pipeline([
        ('enc', TargetEncoder(cols=cat_cols)),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42))
    ])
    X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=(val_size + test_size), random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    results[name] = {
        'MAE': mean_absolute_error(y_te, preds),
        'R2': r2_score(y_te, preds),
        'Test Size': f'{int(test_size*100)}%'
    }

# 5-Fold CV
pipe = Pipeline([
    ('enc', TargetEncoder(cols=cat_cols)),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42))
])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = cross_val_score(pipe, X, y, cv=kf, scoring='r2')
results['5-Fold CV'] = {
    'MAE': None,
    'R2': r2_scores.mean(),
    'Test Size': '20% each'
}

# Plot split comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Split Strategy Comparison (V1 Random Forest)', fontsize=15, fontweight='bold', color='white')

names = list(results.keys())
r2_vals = [results[n]['R2'] for n in names]
mae_vals = [results[n]['MAE'] if results[n]['MAE'] else 0 for n in names]

bar_colors = [ACCENT2 if n == '70-30' or n == '5-Fold CV' else ACCENT for n in names]

ax = axes[0]
bars = ax.bar(names, r2_vals, color=bar_colors, alpha=0.85, edgecolor='#444466')
ax.set_title('R² Score by Split Strategy', fontweight='bold')
ax.set_ylabel('R² Score')
ax.set_ylim(0, 1.0)
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=9, color='white', fontweight='bold')
ax.tick_params(axis='x', rotation=20)
ax.grid(True, axis='y', alpha=0.5)

ax = axes[1]
bars = ax.bar(names, mae_vals, color=bar_colors, alpha=0.85, edgecolor='#444466')
ax.set_title('MAE (MAD) by Split Strategy', fontweight='bold')
ax.set_ylabel('Mean Absolute Error (MAD)')
for bar, val in zip(bars, mae_vals):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{val:,.0f}', ha='center', fontsize=8, color='white', fontweight='bold')
ax.tick_params(axis='x', rotation=20)
ax.grid(True, axis='y', alpha=0.5)

# Highlight annotation
axes[0].annotate('Selected!', xy=(names.index('5-Fold CV'), r2_vals[names.index('5-Fold CV')]),
                 xytext=(names.index('5-Fold CV'), r2_vals[names.index('5-Fold CV')] + 0.08),
                 ha='center', color=ACCENT2, fontsize=10, fontweight='bold')
axes[0].annotate('Selected!', xy=(names.index('70-30'), r2_vals[names.index('70-30')]),
                 xytext=(names.index('70-30'), r2_vals[names.index('70-30')] + 0.08),
                 ha='center', color=ACCENT2, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('../images/report_split_comparison.png', dpi=150, bbox_inches='tight', facecolor='#13131f')
print("  -> Saved: images/report_split_comparison.png")
plt.close()

# =========================================================
# 3. Algorithm Comparison (V1 preprocessing)
# =========================================================
print("Running algorithm comparison...")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=42)

encoder = TargetEncoder(cols=cat_cols)
scaler = StandardScaler()
X_tr_enc = scaler.fit_transform(encoder.fit_transform(X_tr, y_tr))
X_te_enc = scaler.transform(encoder.transform(X_te))

models = {
    'Linear\nRegression': LinearRegression(),
    'Ridge\nRegression': Ridge(),
    'KNN\n(k=5)': KNeighborsRegressor(n_neighbors=5),
    'Gradient\nBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
    'Random\nForest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
}

model_results = {}
for name, model in models.items():
    model.fit(X_tr_enc, y_tr)
    preds = model.predict(X_te_enc)
    model_results[name] = {
        'R2': r2_score(y_te, preds),
        'MAE': mean_absolute_error(y_te, preds),
    }

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Algorithm Comparison on V3 Preprocessed Data (70-30 Split)', fontsize=14, fontweight='bold', color='white')

names = list(model_results.keys())
r2_vals = [model_results[n]['R2'] for n in names]
mae_vals = [model_results[n]['MAE'] for n in names]
bar_colors = [ACCENT2 if 'Forest' in n else ACCENT for n in names]

ax = axes[0]
bars = ax.bar(names, r2_vals, color=bar_colors, alpha=0.85, edgecolor='#444466')
ax.set_title('R² Score by Algorithm', fontweight='bold')
ax.set_ylabel('R² Score')
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', fontsize=9, color='white', fontweight='bold')
ax.grid(True, axis='y', alpha=0.5)

ax = axes[1]
bars = ax.bar(names, [m / 1000 for m in mae_vals], color=bar_colors, alpha=0.85, edgecolor='#444466')
ax.set_title('MAE by Algorithm', fontweight='bold')
ax.set_ylabel('MAE (k MAD)')
for bar, val in zip(bars, mae_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val/1000:.1f}k', ha='center', fontsize=9, color='white', fontweight='bold')
ax.grid(True, axis='y', alpha=0.5)

plt.tight_layout()
plt.savefig('../images/report_algorithm_comparison.png', dpi=150, bbox_inches='tight', facecolor='#13131f')
print("  -> Saved: images/report_algorithm_comparison.png")
plt.close()

print("\n✅ All report images generated successfully!")
