# main.py

import pandas as pd
import numpy as np
import zipfile
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    completeness_score,
    homogeneity_score,
    v_measure_score,
    adjusted_rand_score
)

# ============================================================
# 1. LOAD DATA
# ============================================================

with zipfile.ZipFile('data/asteroid.zip', 'r') as zip_ref:
    zip_ref.extractall('data/')

df_asteroid = pd.read_fwf(
    "data/asteroid_data.csv",
    sep=r"\s{2,}",
    skiprows=1,
    engine="python"
)

df_family = pd.read_fwf(
    "data/asteroids_family.csv",
    sep=r"\s{2,}",
    engine="python"
)

df_asteroid = df_asteroid[
    df_asteroid['%Name'].astype(str).str.fullmatch(r'\d+')
]

df_family = df_family[
    df_family['%ast.name'].astype(str).str.fullmatch(r'\d+')
]

df_family = df_family[['%ast.name', 'family1']]

df = pd.merge(
    df_asteroid,
    df_family,
    left_on='%Name',
    right_on='%ast.name',
    how='inner'
)

df = df.drop(columns=[
    '%ast.name', 'mag.', 'n (deg/yr)',
    'g ("/yr)', 's ("/yr)', 'LCEx1E6', 'My'
])

# ============================================================
# 2. SET BEST GROUP HERE
# ============================================================

best_group = [1911, 3, 31, 480, 3330, 569, 10955, 293] #best group for gmm (avg: 54%)

features = ['a (AU)', 'e', 'sin I']

df_dataset = df[df["family1"].isin(best_group)].copy()
df_dataset["family1"] = df_dataset["family1"].astype(str)

X_raw = df_dataset[features]
y = df_dataset["family1"]

print(f"\nDataset size: {len(df_dataset)}")

# ============================================================
# 3. NORMALIZATION
# ============================================================

scaler = RobustScaler()
X = scaler.fit_transform(X_raw)

# ============================================================
# 4. GMM
# ============================================================

gmm = GaussianMixture(
    n_components=8,
    covariance_type='full',
    n_init=20,
    random_state=42
)

labels = gmm.fit_predict(X)
labels_str = np.array([f"C{i}" for i in labels])

# ============================================================
# 5. EVALUATION
# ============================================================

comp = completeness_score(y, labels_str)
homo = homogeneity_score(y, labels_str)
vm = v_measure_score(y, labels_str)
ari = adjusted_rand_score(y, labels_str)

print("\nOverall Metrics:")
print(f"Completeness: {comp:.4f}")
print(f"Homogeneity:  {homo:.4f}")
print(f"V-measure:    {vm:.4f}")
print(f"ARI:          {ari:.4f}")

print("\nPer-family completeness:\n")

fam_comp = []

for fam in map(str, best_group):
    mask = y == fam
    counts = pd.Series(labels_str[mask]).value_counts()
    pct = (counts.iloc[0] / mask.sum()) * 100
    fam_comp.append(pct)
    print(f"{fam}: {pct:.1f}%")

print(f"\nAvg Family Completeness: {np.mean(fam_comp):.1f}%")