from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import zipfile
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from sklearn.preprocessing import StandardScaler, MinMaxScaler

with zipfile.ZipFile('data/asteroid.zip', 'r') as zip_ref:
    zip_ref.extractall('data/')

df_asteroid = pd.read_fwf( #fixed witdh file, with regex separator to handle multiple spaces, and python engine to support regex separator
    "data/asteroid_data.csv",
    sep=r"\s{2,}",
    skiprows=1,
    engine="python"
)

df_asteroid_family = pd.read_fwf(
    "data/asteroids_family.csv",
    sep=r"\s{2,}",
    engine="python"
)

#only numbered asteroids
df_asteroid_clean = df_asteroid[
    df_asteroid['%Name'].astype(str).str.fullmatch(r'\d+')
]

df_family_clean = df_asteroid_family[
    df_asteroid_family['%ast.name'].astype(str).str.fullmatch(r'\d+')
]
df_family_clean = df_family_clean[['%ast.name', 'family1']]

# merge data frames
df_merged = pd.merge(
    df_asteroid_clean,
    df_family_clean,
    left_on='%Name',
    right_on='%ast.name',
    how='inner'
)

df_merged = df_merged.drop(columns=['%ast.name'])

#search for missing values
print(df_merged.info())
print(df_merged.isna().sum()) #there are no missing values

quantitative_columns = [
    col for col in df_merged.columns
    if col not in ["%Name", "family1", "My"]  # exclude id, categorical, and My (only 3 unique values)
]

recommendations = {}
X = df_merged[quantitative_columns].copy()

# Decide and apply transformations in one loop
for col in quantitative_columns:
    data = df_merged[col].values.ravel()

    if np.mean(data) == 0:
        cv = np.inf
    else:
        cv = (np.std(data) / np.mean(data)) * 100

    needs_log = cv > 100
    
    if needs_log:
        min_val = data.min()
        if min_val <= 0:
            shift = abs(min_val) + 1e-9
            data_transformed = np.log(data + shift)
        else:
            data_transformed = np.log(data)
        
        try:
            stat_log, p_value_log = normaltest(data_transformed)
        except Exception:
            p_value_log = 0.0
        
        if p_value_log > 0.05:
            recommendations[col] = ("Log+Z-score", {"p_value_original": None, "p_value_log": float(p_value_log), "cv": cv, "needs_log": True})
            scaler = StandardScaler()
            X[col] = scaler.fit_transform(data_transformed.reshape(-1, 1)).ravel()
        else:
            recommendations[col] = ("Log+MinMax", {"p_value_original": None, "p_value_log": float(p_value_log), "cv": cv, "needs_log": True})
            scaler = MinMaxScaler()
            X[col] = scaler.fit_transform(data_transformed.reshape(-1, 1)).ravel()
    else:
        try:
            stat, p_value = normaltest(data)
        except Exception:
            p_value = 0.0
        
        if p_value > 0.05:
            recommendations[col] = ("Z-score", {"p_value_original": float(p_value), "p_value_log": None, "cv": cv, "needs_log": False})
            scaler = StandardScaler()
            X[col] = scaler.fit_transform(data.reshape(-1, 1)).ravel()
        else:
            recommendations[col] = ("MinMax", {"p_value_original": float(p_value), "p_value_log": None, "cv": cv, "needs_log": False})
            scaler = MinMaxScaler()
            X[col] = scaler.fit_transform(data.reshape(-1, 1)).ravel()

# Plot original vs transformed
n_rows = len(quantitative_columns)
n_cols = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.2 * n_rows))
axes = axes.reshape(n_rows, n_cols)

for i, col in enumerate(quantitative_columns):
    # ORIGINAL
    data_orig = df_merged[col].dropna().values.ravel()
    axes[i, 0].hist(
        data_orig,
        bins=40,
        density=True,
        color='steelblue',
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85
    )
    details = recommendations[col][1]
    p_orig = details.get('p_value_original', None)
    p_log = details.get('p_value_log', None)
    needs_log = details.get('needs_log', False)
    
    if needs_log and p_log is not None:
        p_text = f" (p_log={p_log:.4f})"
    elif p_orig is not None:
        p_text = f" (p={p_orig:.4f})"
    else:
        p_text = ""
    
    axes[i, 0].set_title(f"{col} (Original){p_text}", fontsize=9)
    axes[i, 0].tick_params(axis='x', labelsize=7)

    # TRANSFORMED (reuse already transformed data)
    data_t = X[col].values
    rec = recommendations[col][0]
    title = f"{col} ({rec})"
    
    axes[i, 1].hist(
        data_t,
        bins=40,
        density=True,
        color='mediumseagreen',
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85
    )
    axes[i, 1].set_title(title, fontsize=9)
    axes[i, 1].tick_params(axis='x', labelsize=7)

plt.tight_layout()
plt.savefig("plots/original_vs_recommended.png", dpi=150)
plt.close()

y = df_merged['family1']

# Plot class distribution
class_counts = y.value_counts().sort_values(ascending=False)
plt.figure(figsize=(14, 6))
plt.bar(range(len(class_counts)), class_counts.values, color='steelblue', edgecolor='black', linewidth=0.5)
plt.xlabel('Class Index (sorted by frequency)', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.title(f'Class Distribution - {len(class_counts)} classes (total: {len(y)} samples)', fontsize=14, fontweight='bold')
plt.yscale('log') 
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig("plots/class_distribution.png", dpi=150)
plt.close()

# Filter classes with <2 samples (needed for stratify)
mask = y.isin(class_counts[class_counts >= 2].index)
X_filtered = X[mask]
y_filtered = y[mask]
print(f"\nFiltered {(~mask).sum()} samples from classes with <2 samples")

# INSANE Class disbalance, possible cause of error
# Split: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.3, random_state=42, stratify=y_filtered)

clf = MLPClassifier(
    hidden_layer_sizes=(150, 100, 70, 50, 20), 
    max_iter=500, 
    random_state=42, 
    verbose=True,
    early_stopping=True,
    n_iter_no_change=15,
    validation_fraction=0.1, #use 10% of training data for early stopping validation
    learning_rate='adaptive', 
    learning_rate_init=0.005   
)
clf.fit(X_train, y_train)

y_test_pred = clf.predict(X_test)
print(f"\n=== Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f} ===")
print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred, zero_division=0))

""""ROUGHT IMPLEMENTATION, I only need to classify 8 families, still need to discover wich ones are the easiest and to them"""

"""Clustering could be better ?"""