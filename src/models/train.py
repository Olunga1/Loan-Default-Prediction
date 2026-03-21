from __future__ import annotations

import logging
from pathlib import Path

import hydra
import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data.ingestion import DataIngestion
from src.data.preprocessing import FeaturePreprocessor
from src.utils.config import setup_logging


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="training")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging)

    ingestion = DataIngestion(cfg.data)
    data_path = cfg.data.train_path
    if not Path(data_path).exists() and Path("Loan_default.csv").exists():
        data_path = "Loan_default.csv"

    df = ingestion.load_data(data_path, source_type="csv", validate=False)

    target_col = "Default" if "Default" in df.columns else "default"
    drop_candidates = {target_col, "LoanID", "loan_id"}
    feature_cols = [c for c in df.columns if c not in drop_candidates]

    X = df[feature_cols]
    y = df[target_col]

    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = FeaturePreprocessor(categorical_cols=categorical_cols, numerical_cols=numerical_cols)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.validation_split, random_state=cfg.random_state, stratify=y
    )

    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=cfg.random_state,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    model.fit(X_train_p, y_train)

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val_p)[:, 1])
    logger.info("Validation AUC: %.4f", val_auc)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name="baseline_train"):
        mlflow.log_param("model", "xgboost")
        mlflow.log_metric("validation_auc", float(val_auc))

    out_dir = Path(cfg.model.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.pkl")
    joblib.dump(preprocessor, out_dir / "preprocessor.pkl")


if __name__ == "__main__":
    main()
