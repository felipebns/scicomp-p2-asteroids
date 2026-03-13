import pandas as pd
import numpy as np
import zipfile
import plotly.express as px
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture

class Model:
    def __init__(self, groups: list, covariance_type: str, n_init: int) -> None:
        self._groups = groups
        self._covariance_type = covariance_type
        self._n_init = n_init

    @staticmethod
    def _read_fwf_path(path: str, skiprows: int) -> pd.DataFrame:
        return pd.read_fwf(
            path,
            sep=r"\s{2,}",
            skiprows=skiprows,
            engine="python"
        )

    @staticmethod
    def _merge_dfs(df_asteroid: pd.DataFrame, df_family: pd.DataFrame) -> pd.DataFrame:
        df_asteroid = df_asteroid[df_asteroid['%Name'].astype(str).str.fullmatch(r'\d+')]
        df_family = df_family[df_family['%ast.name'].astype(str).str.fullmatch(r'\d+')]
        df_family = df_family[['%ast.name', 'family1']]
        df = pd.merge(
            df_asteroid,
            df_family,
            left_on='%Name',
            right_on='%ast.name',
            how='inner'
        )
        
        return df
    
    @staticmethod
    def _normalize(X_raw: pd.DataFrame) -> np.ndarray:
        scaler = RobustScaler()
        X = scaler.fit_transform(X_raw)
        return X
    
    def get_data(self) -> pd.DataFrame:
        with zipfile.ZipFile('data/asteroid.zip', 'r') as zip_ref:
            zip_ref.extractall('data/')

        df_asteroid = self._read_fwf_path("data/asteroid_data.csv", skiprows=1)
        df_family = self._read_fwf_path("data/asteroids_family.csv", skiprows=0)

        df = self._merge_dfs(df_asteroid, df_family)

        df = df.drop(columns=[
            '%ast.name', 'mag.', 'n (deg/yr)',
            'g ("/yr)', 's ("/yr)', 'LCEx1E6', 'My'
        ])

        return df
    
    def get_features_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        features = ['a (AU)', 'e', 'sin I']

        df_dataset = df[df["family1"].isin(self._groups)].copy()
        df_dataset["family1"] = df_dataset["family1"].astype(str)

        X_raw = df_dataset[features]
        y = df_dataset["family1"]

        return X_raw, y
    
    def get_clustering_results(self, X_raw: np.ndarray) -> np.ndarray:
        X = self._normalize(X_raw)
        gmm = GaussianMixture(
            n_components=len(self._groups),
            covariance_type=self._covariance_type,
            n_init=self._n_init,
            random_state=42
        )

        results = gmm.fit_predict(X)
        results_str = np.array([f"C{i}" for i in results])

        return results_str
    
    def evaluate_clustering(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        fam_comp = []
        groups_str = [str(g) for g in self._groups]

        for fam in groups_str:
            mask = y_true == fam
            counts = pd.Series(y_pred[mask]).value_counts()
            pct = (counts.iloc[0] / mask.sum()) * 100
            fam_comp.append(pct)
            print(f"{fam}: {pct:.1f}%")

        print(f"\nAvg Family Completeness: {np.mean(fam_comp)}%")

    def plot_3d_real_vs_pred(self, X_raw: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray) -> None:
        df_plot = pd.DataFrame(X_raw).copy()
        df_plot.columns = ['a (AU)', 'e', 'sin I']
        df_plot['Real'] = np.array(y_true).astype(str)
        df_plot['Predict'] = np.array(y_pred).astype(str)

        fig_real = px.scatter_3d(
            df_plot,
            x='a (AU)',
            y='e',
            z='sin I',
            color='Real',
            title='Real (3D)',
            opacity=0.8
        )

        fig_pred = px.scatter_3d(
            df_plot,
            x='a (AU)',
            y='e',
            z='sin I',
            color='Predict',
            title='Predict (3D)',
            opacity=0.8
        )

        axis_ranges = {
            'x': [df_plot['a (AU)'].min(), df_plot['a (AU)'].max()],
            'y': [df_plot['e'].min(), df_plot['e'].max()],
            'z': [df_plot['sin I'].min(), df_plot['sin I'].max()]
        }

        fig_real.update_layout(
            template='plotly_white',
            scene={
                'xaxis': {'title': 'a (AU)', 'range': axis_ranges['x']},
                'yaxis': {'title': 'e', 'range': axis_ranges['y']},
                'zaxis': {'title': 'sin I', 'range': axis_ranges['z']}
            }
        )
        fig_pred.update_layout(
            template='plotly_white',
            scene={
                'xaxis': {'title': 'a (AU)', 'range': axis_ranges['x']},
                'yaxis': {'title': 'e', 'range': axis_ranges['y']},
                'zaxis': {'title': 'sin I', 'range': axis_ranges['z']}
            }
        )

        fig_real.show()
        fig_pred.show()