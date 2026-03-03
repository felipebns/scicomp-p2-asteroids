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

# Perform data cleaning, maintaining columns, normalizing, removing invalid values and keeping only numbered asteroids!

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
# print(len(df_merged))
# print(df_merged.head())

#search for missing values
print(df_merged.info())
print(df_merged.isna().sum()) #there are no missing values

# ----------------------------
# columns to analyze
# ----------------------------
quantitative_columns = [
    col for col in df_merged.columns
    if col not in ["%Name", "family1", "My"]  # exclude id, categorical, and My (only 3 unique values)
]

# ----------------------------
# Decide recommendation per variable
# ----------------------------
recommendations = {}  # col -> ("Z-score" | "MinMax" | "Log+Z-score" | "Log+MinMax", details dict)
outlier_analysis = {}  # col -> outlier metrics

for col in quantitative_columns:
    data = df_merged[col].values.ravel()
    # if constant or too small, fallback to MinMax
    if data.size < 8 or np.allclose(data, data[0]):
        recommendations[col] = ("MinMax", {"reason": "constant_or_too_small", "p_value": None})
        outlier_analysis[col] = {"n_outliers": 0, "pct_outliers": 0.0}
        continue

    # Outlier analysis using IQR method
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = (data < lower_bound) | (data > upper_bound)
    n_outliers = np.sum(outliers)
    pct_outliers = (n_outliers / len(data)) * 100
    
    # Calculate outlier intensity metrics
    max_val = np.max(data)
    min_val = np.min(data)
    outlier_ratio_upper = (max_val / q3) if q3 != 0 else np.inf
    outlier_ratio_lower = (q1 / min_val) if min_val != 0 else np.inf
    
    # Coefficient of variation (std/mean) - higher = more dispersion
    cv = (np.std(data) / np.mean(data)) * 100 if np.mean(data) != 0 else np.inf
    
    outlier_analysis[col] = {
        "n_outliers": n_outliers,
        "pct_outliers": pct_outliers,
        "outlier_ratio_upper": outlier_ratio_upper,
        "outlier_ratio_lower": outlier_ratio_lower,
        "cv": cv
    }

    # Decide if log transform is needed based on outlier metrics
    needs_log = (cv > 100 or outlier_ratio_upper > 3)
    
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
        # No log transform needed, test normality on original data
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

# print summary
print("\nVariable\t\tp_value_orig\tp_value_log\tCV(%)\t\tRecommendation")
print("=" * 90)
for col in quantitative_columns:
    info = recommendations[col]
    details = info[1]
    p_orig = details.get('p_value_original', None)
    p_log = details.get('p_value_log', None)
    cv = details.get('cv', 'N/A')
    
    p_orig_str = f"{p_orig:.6f}" if p_orig is not None else "N/A"
    p_log_str = f"{p_log:.6f}" if p_log is not None else "N/A"
    cv_str = f"{cv:.2f}" if isinstance(cv, (int, float)) and cv != np.inf else "N/A"
    
    print(f"{col:20}\t{p_orig_str}\t{p_log_str}\t{cv_str:10}\t{info[0]}")

# print outlier analysis
print("\n" + "=" * 90)
print("OUTLIER ANALYSIS (IQR Method: Q1-1.5*IQR to Q3+1.5*IQR)")
print("=" * 90)
print(f"{'Variable':<20} {'N_Outliers':<12} {'%_Outliers':<12} {'CV(%)':<10} {'Max/Q3':<10} {'Q1/Min':<10}")
print("-" * 90)
for col in quantitative_columns:
    oa = outlier_analysis[col]
    n_out = oa['n_outliers']
    pct_out = oa['pct_outliers']
    cv = oa['cv']
    ratio_upper = oa['outlier_ratio_upper']
    ratio_lower = oa['outlier_ratio_lower']
    
    cv_str = f"{cv:.2f}" if cv != np.inf else "inf"
    ratio_upper_str = f"{ratio_upper:.2f}" if ratio_upper != np.inf else "inf"
    ratio_lower_str = f"{ratio_lower:.2f}" if ratio_lower != np.inf else "inf"
    
    print(f"{col:<20} {n_out:<12} {pct_out:<12.2f} {cv_str:<10} {ratio_upper_str:<10} {ratio_lower_str:<10}")

print("\nLegend:")
print("- N_Outliers: Number of values outside 1.5*IQR range")
print("- %_Outliers: Percentage of outliers")
print("- CV(%): Coefficient of Variation (std/mean * 100) - higher = more dispersion")
print("- Max/Q3: Ratio of max value to 3rd quartile - high value indicates extreme upper outliers")
print("- Q1/Min: Ratio of 1st quartile to min value - high value indicates extreme lower outliers")
print("\nVariables with CV > 100% or Max/Q3 > 3 may benefit from log transformation")

# ----------------------------
# Plot: Original vs recommended transform (grid, 2 columns)
# ----------------------------
n_rows = len(quantitative_columns)
n_cols = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.2 * n_rows))
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

print("Saved plot: plots/original_vs_recommended.png")

# ----------------------------
# Plot: Boxplot comparison (Original vs Normalized)
# ----------------------------
fig_box, axes_box = plt.subplots(n_rows, n_cols, figsize=(10, 2.2 * n_rows))
axes_box = axes_box.reshape(n_rows, n_cols)

for i, col in enumerate(quantitative_columns):
    data = df_merged[col].dropna().values.ravel()
    
    # ORIGINAL BOXPLOT
    bp1 = axes_box[i, 0].boxplot(
        [data],
        vert=True,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        flierprops=dict(marker='o', markersize=3, alpha=0.5)
    )
    for patch in bp1['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    
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
    
    axes_box[i, 0].set_title(f"{col} (Original){p_text}", fontsize=9)
    axes_box[i, 0].set_ylabel("Value", fontsize=7)
    axes_box[i, 0].tick_params(axis='y', labelsize=7)
    axes_box[i, 0].set_xticklabels([])
    
    # NORMALIZED BOXPLOT
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
    
    bp2 = axes_box[i, 1].boxplot(
        [data_t],
        vert=True,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        flierprops=dict(marker='o', markersize=3, alpha=0.5)
    )
    for patch in bp2['boxes']:
        patch.set_facecolor('mediumseagreen')
        patch.set_alpha(0.7)
    
    axes_box[i, 1].set_title(title, fontsize=9)
    axes_box[i, 1].set_ylabel("Value", fontsize=7)
    axes_box[i, 1].tick_params(axis='y', labelsize=7)
    axes_box[i, 1].set_xticklabels([])

plt.tight_layout()
plt.savefig("plots/boxplot_comparison.png", dpi=150)
plt.close()

print("Saved plot: plots/boxplot_comparison.png")