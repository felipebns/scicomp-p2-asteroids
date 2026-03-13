from model import Model

best_group = [1911, 31, 3, 410, 3330, 293, 1298, 12739]
model = Model(groups=best_group, covariance_type='full', n_init=10)
data = model.get_data()
X_raw, y = model.get_features_target(data)
results = model.get_clustering_results(X_raw)
model.evaluate_clustering(y, results)
model.plot_3d_real_vs_pred(X_raw, y, results)