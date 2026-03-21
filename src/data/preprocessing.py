from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeaturePreprocessor:
    def __init__(self, categorical_cols: list[str], numerical_cols: list[str]):
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[("scaler", StandardScaler())]), numerical_cols),
                ("cat", Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
            ]
        )

    def fit(self, X):
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)

    def fit_transform(self, X):
        return self.pipeline.fit_transform(X)

    def get_feature_names_out(self):
        return self.pipeline.get_feature_names_out()
