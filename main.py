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

recommendations = {}  # col -> ("Z-score" | "MinMax" | "Log+Z-score" | "Log+MinMax", details dict)

for col in quantitative_columns:
    data = df_merged[col].values.ravel()

    if np.mean(data) == 0:
        cv = np.inf  # infinite CV if mean is zero (extreme case)
    else:
        cv = (np.std(data) / np.mean(data)) * 100

    # Decide if log transform is needed based on CV
    needs_log = cv > 100 # if std is much bigger than mean, we have a very skewed distribution, and log can help reduce skewness and stabilize variance
    
    if needs_log:
        # Apply log transform first (handle non-positive values)
        min_val = data.min()
        if min_val <= 0:
            shift = abs(min_val) + 1e-9
            data_transformed = np.log(data + shift)
        else:
            data_transformed = np.log(data)
        
        # Test normality on log-transformed data
        try:
            stat_log, p_value_log = normaltest(data_transformed)
        except Exception:
            p_value_log = 0.0
        
        # Decide normalization method on log-transformed data
        if p_value_log > 0.05:
            recommendations[col] = ("Log+Z-score", {
                "p_value_original": None,
                "p_value_log": float(p_value_log),
                "cv": cv,
                "needs_log": True
            })
        else:
            recommendations[col] = ("Log+MinMax", {
                "p_value_original": None,
                "p_value_log": float(p_value_log),
                "cv": cv,
                "needs_log": True
            })
    else:
        try:
            stat, p_value = normaltest(data)
        except Exception:
            p_value = 0.0
        
        # Decision: use alpha=0.05 (standard statistical threshold)
        if p_value > 0.05:
            recommendations[col] = ("Z-score", {
                "p_value_original": float(p_value),
                "p_value_log": None,
                "cv": cv,
                "needs_log": False
            })
        else:
            recommendations[col] = ("MinMax", {
                "p_value_original": float(p_value),
                "p_value_log": None,
                "cv": cv,
                "needs_log": False
            })

n_rows = len(quantitative_columns)
n_cols = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2.2 * n_rows))
axes = axes.reshape(n_rows, n_cols)

for i, col in enumerate(quantitative_columns):
    data = df_merged[col].dropna().values.ravel()

    # ORIGINAL
    axes[i, 0].hist(
        data,
        bins=40,
        density=True,
        color='steelblue',
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85
    )
    # Get p-value for title
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

    # TRANSFORMED according to recommendation
    rec = recommendations[col][0]
    if rec == "Z-score":
        scaler = StandardScaler()
        data_t = scaler.fit_transform(data.reshape(-1, 1)).ravel()
        title = f"{col} (Z-score)"
    elif rec == "Log+Z-score":
        # Apply log first, then Z-score
        min_val = data.min()
        if min_val <= 0:
            shift = abs(min_val) + 1e-9
            data_log = np.log(data + shift)
        else:
            data_log = np.log(data)
        scaler = StandardScaler()
        data_t = scaler.fit_transform(data_log.reshape(-1, 1)).ravel()
        title = f"{col} (Log+Z-score)"
    elif rec == "Log+MinMax":
        # Apply log first, then MinMax
        min_val = data.min()
        if min_val <= 0:
            shift = abs(min_val) + 1e-9
            data_log = np.log(data + shift)
        else:
            data_log = np.log(data)
        scaler = MinMaxScaler()
        data_t = scaler.fit_transform(data_log.reshape(-1, 1)).ravel()
        title = f"{col} (Log+MinMax)"
    else:  # MinMax
        scaler = MinMaxScaler()
        data_t = scaler.fit_transform(data.reshape(-1, 1)).ravel()
        title = f"{col} (MinMax)"

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

# clean layout and save
plt.tight_layout()
plt.savefig("plots/original_vs_recommended.png", dpi=150)
plt.close()
