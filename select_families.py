# select_families.py

import pandas as pd
import numpy as np
import zipfile
from itertools import combinations
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture

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

features = ['a (AU)', 'e', 'sin I']

# ============================================================
# 2. FILTER BY SIZE
# ============================================================

MIN_SIZE = 300
MAX_SIZE = 2000

family_counts = df['family1'].value_counts()

valid_families = family_counts[
    (family_counts >= MIN_SIZE) &
    (family_counts <= MAX_SIZE)
].index.tolist()

print(f"Valid families: {len(valid_families)}")

# ============================================================
# 3. BHATTACHARYYA-BASED SEPARABILITY SCORE
# ============================================================

from numpy.linalg import det, inv

scaler = RobustScaler()
X_all = scaler.fit_transform(df[features])
X_all = pd.DataFrame(X_all, columns=features)

means = {}
covs = {}

for fam in valid_families:
    mask = df['family1'] == fam
    X_f = X_all[mask].values

    means[fam] = X_f.mean(axis=0)
    cov = np.cov(X_f, rowvar=False)

    # Regularização para evitar singularidade
    cov += 1e-6 * np.eye(cov.shape[0])

    covs[fam] = cov

def bhattacharyya_distance(mu1, mu2, S1, S2):
    S = 0.5 * (S1 + S2)
    diff = mu1 - mu2

    term1 = 0.125 * diff.T @ inv(S) @ diff
    term2 = 0.5 * np.log(det(S) / np.sqrt(det(S1) * det(S2)))

    return term1 + term2

scores = {}

for fam in valid_families:

    distances = []

    for other in valid_families:
        if fam == other:
            continue

        d = bhattacharyya_distance(
            means[fam],
            means[other],
            covs[fam],
            covs[other]
        )

        distances.append(d)

    # usamos a menor distância (caso mais crítico)
    min_db = min(distances)

    scores[fam] = min_db

# Selecionar top 15 famílias mais separadas
top_candidates = sorted(scores, key=scores.get, reverse=True)[:15]

print("Top candidates (Bhattacharyya separability):")
for fam in top_candidates:
    print(f"Family {fam}: size={family_counts[fam]}, score={scores[fam]:.4f}")

# ============================================================
# 4. TEST COMBINATIONS OF 8
# ============================================================

best_group = None
best_score = -np.inf
best_metrics = None

for group in combinations(top_candidates, 8):

    df_subset = df[df["family1"].isin(group)].copy()
    df_subset["family1"] = df_subset["family1"].astype(str)

    X_raw = df_subset[features]
    y = df_subset["family1"]

    X_scaled = scaler.fit_transform(X_raw)

    gmm = GaussianMixture(
        n_components=8,
        covariance_type='full',
        n_init=10,
        random_state=42
    )

    labels = gmm.fit_predict(X_scaled)
    labels_str = np.array([f"C{i}" for i in labels])

    fam_comp = []

    for fam in map(str, group):
        mask = y == fam
        counts = pd.Series(labels_str[mask]).value_counts()
        pct = (counts.iloc[0] / mask.sum()) * 100
        fam_comp.append(pct)

    avg_family = np.mean(fam_comp)
    min_family = np.min(fam_comp)

    score = avg_family + 0.5 * min_family

    if score > best_score:
        best_score = score
        best_group = group
        best_metrics = (avg_family, min_family)

    print(f"{group} → avg={avg_family:.1f}% min={min_family:.1f}%")

print("\n==============================")
print("BEST GROUP FOUND:")
print(best_group)
print("Avg completeness:", best_metrics[0])
print("Min completeness:", best_metrics[1])