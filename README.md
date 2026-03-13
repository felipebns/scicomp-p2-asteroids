# Asteroid Family Clustering (GMM)

This project clusters asteroid families from proper elements and evaluates family completeness.

Data source: [AstDyS](https://newton.spacedys.com/astdys2/index.php?pc=1.0.0) (numbered objects only).

## How to run

1. Create and activate a virtual environment.
2. Install dependencies:

	pip install -r requirements.txt

3. Run:

	python main.py

## Benchmark comparison (95% completeness per family)

The benchmark is 95% completeness for each family.

| Family | Completeness (%) | Meets 95% benchmark |
|---|---:|:---:|
| 1911 | 44.3 | No |
| 31 | 39.0 | No |
| 3 | 65.6 | No |
| 410 | 58.2 | No |
| 3330 | 58.0 | No |
| 293 | 43.8 | No |
| 1298 | 40.8 | No |
| 12739 | 42.1 | No |

Average family completeness: **48.9950%**

## Why GMM for asteroid clustering

`GaussianMixture` was selected because asteroid families in proper-element space are often anisotropic and can overlap. GMM is a good fit because it:
- models elliptical clusters via full covariance;
- provides soft probabilistic modeling before hard assignment;
- handles unequal cluster variances better than centroid-only methods.

### GMM vs Hierarchical Clustering

**GMM advantages in this project**
- Supports overlapping families through probabilistic assignment.
- Fits anisotropic/elliptical shapes with covariance matrices.
- Lets us explicitly set `n_components` to the expected number of families.
- Is straightforward to re-fit many times during family-combination search.

**Hierarchical clustering limitations here**
- Hard assignments only (no class probabilities).
- Sensitive to linkage and distance choices.
- Can struggle when cluster densities and variances differ significantly.
- Produces a dendrogram that still requires a cut rule, which can be unstable across subsets.


## Family selection pipeline

The class `FamilySelector` in [select_families.py](select_families.py) performs the slow search for stronger family sets:

1. Filters valid families by size (`min_size` to `max_size`).
2. Computes the`Bhattacharyya distance` between family Gaussian approximations.
3. Ranks families by their worst-case (minimum) separation.
4. Keeps the top candidates (`top_k_candidates`).
5. Tests combinations of `group_size` families with the model pipeline.
6. Scores each combination with:

	$$
		ext{score} = \text{avg completeness} + 0.5 \times \text{min completeness}
	$$

7. Returns the best group and summary metrics.

### What the `Bhattacharyya distance` means

`Bhattacharyya distance` measures how separated two probability distributions are.

In this project, each family is approximated by a Gaussian with mean $\mu$ and covariance $\Sigma$ in the feature space (`a (AU)`, `e`, `sin I`).

For two families, the distance combines:
- a **mean-separation term** (how far the centers are), and
- a **covariance-overlap term** (how similar/spread their shapes are).

Interpretation:
- higher value means stronger separation (easier to distinguish),
- lower value means more overlap/confusion risk.

The selector uses each family's **minimum** distance to all others (worst case) to rank robustness.

## How to visualize results

After `python main.py`, two Plotly windows/tabs are shown:
- Real (3D): color by ground-truth family label.
- Predict (3D): color by GMM cluster label.

Axes are:
- `a (AU)`
- `e`
- `sin I`

You can rotate, zoom, and pan to inspect overlap and separability.

## Optional long-running family selection

In [main.py](main.py), there is a commented block showing how to run `FamilySelector`.
It is intentionally disabled by default because it can take a long time.