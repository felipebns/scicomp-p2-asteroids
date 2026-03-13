from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from numpy.linalg import det, inv

from model import Model


class FamilySelector:
    def __init__(self, model: Model, min_size: int = 300, max_size: int = 2000, top_k_candidates: int = 15, group_size: int = 8) -> None:
        self._model = model
        self._min_size = min_size
        self._max_size = max_size
        self._top_k_candidates = top_k_candidates
        self._group_size = group_size

    @staticmethod
    def _bhattacharyya_distance(mu1: np.ndarray, mu2: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> float:
        s = 0.5 * (s1 + s2)
        diff = mu1 - mu2

        term1 = 0.125 * diff.T @ inv(s) @ diff
        term2 = 0.5 * np.log(det(s) / np.sqrt(det(s1) * det(s2)))

        return float(term1 + term2)

    @staticmethod
    def _family_completeness(y_true: pd.Series, y_pred: np.ndarray, group: list[int]) -> tuple[float, float]:
        fam_comp = []
        groups_str = [str(g) for g in group]
        
        for fam in groups_str:
            mask = y_true == fam
            counts = pd.Series(y_pred[mask]).value_counts()
            pct = (counts.iloc[0] / mask.sum()) * 100
            fam_comp.append(pct)

        avg_family = float(np.mean(fam_comp))
        min_family = float(np.min(fam_comp))

        return avg_family, min_family

    def _get_valid_families(self, df: pd.DataFrame) -> list[Any]:
        family_counts = df['family1'].value_counts()
        valid_families = family_counts[
            (family_counts >= self._min_size) &
            (family_counts <= self._max_size)
        ].index.tolist()

        return valid_families

    def _get_top_candidates(self, df: pd.DataFrame, valid_families: list[Any]) -> list[Any]:
        features = ['a (AU)', 'e', 'sin I']
        X_all = self._model._normalize(df[features])
        X_all = pd.DataFrame(X_all, columns=features)

        means = {}
        covs = {}

        for fam in valid_families:
            mask = df['family1'] == fam
            X_f = X_all[mask].values

            means[fam] = X_f.mean(axis=0)
            cov = np.cov(X_f, rowvar=False)
            cov += 1e-6 * np.eye(cov.shape[0])
            covs[fam] = cov

        scores = {}

        for fam in valid_families:
            distances = []

            for other in valid_families:
                if fam == other:
                    continue

                d = self._bhattacharyya_distance(
                    means[fam],
                    means[other],
                    covs[fam],
                    covs[other]
                )
                distances.append(d)

            scores[fam] = min(distances)

        top_candidates = sorted(scores, key=scores.get, reverse=True)[:self._top_k_candidates]
        return top_candidates

    def find_best_group(self) -> dict[str, Any]:
        df = self._model.get_data()
        valid_families = self._get_valid_families(df)
        top_candidates = self._get_top_candidates(df, valid_families)

        best_group = None
        best_score = -np.inf
        best_metrics = None

        for group in combinations(top_candidates, self._group_size):
            original_groups = self._model._groups
            self._model._groups = list(group)

            X_raw, y = self._model.get_features_target(df)
            y_pred = self._model.get_clustering_results(X_raw)

            self._model._groups = original_groups

            avg_family, min_family = self._family_completeness(y, y_pred, group)
            score = avg_family + 0.5 * min_family

            if score > best_score:
                best_score = score
                best_group = group
                best_metrics = (avg_family, min_family)

            print(f"{group} -> avg={avg_family:.1f}% min={min_family:.1f}%")

        return {
            'best_group': best_group,
            'avg_completeness': best_metrics[0] if best_metrics else None,
            'min_completeness': best_metrics[1] if best_metrics else None,
            'score': float(best_score),
            'top_candidates': top_candidates,
            'valid_families_count': len(valid_families),
        }