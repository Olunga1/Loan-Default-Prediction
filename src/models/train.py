"""
Model training script with Hydra configuration management.
Supports hyperparameter optimization, experiment tracking, and model versioning.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
import joblib
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.ingestion import DataIngestion
from src.data.preprocessing import FeaturePreprocessor
from src.utils.config import setup_logging

logger = logging.getLogger(__name__)

class LoanDefaultTrainer:
    """Handles model training with experiment tracking and hyperparameter optimization."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.model = None
        self.preprocessor = None
        self.experiment_id = None
        
        # Setup MLflow
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)
        
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare training data."""
        logger.info("Loading training data...")
        
        # Initialize data ingestion
        ingestion = DataIngestion(self.config.data)
        
        # Load data
        train_df = ingestion.load_data(
            self.config.data.train_path,
            source_type='csv',
            validate=True
        )
        
        # Separate features and target
        X = train_df.drop(['LoanID', 'Default'], axis=1)
        y = train_df['Default']
        
        logger.info(f"Loaded {len(X)} training samples")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def create_preprocessor(self) -> ColumnTransformer:
        """Create preprocessing pipeline for features."""
        # Identify categorical and numerical columns
        categorical_cols = [
            'Education', 'EmploymentType', 'MaritalStatus', 
            'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
        ]
        numerical_cols = [
            'Age', 'Income', 'LoanAmount', 'CreditScore', 
            'MonthsEmployed', 'NumCreditLines', 'InterestRate', 
            'LoanTerm', 'DTIRatio'
        ]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        return preprocessor
    
    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Optuna objective function for hyperparameter optimization."""
        
        # Define hyperparameter search space
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
            'random_state': self.config.random_state
        }
        
        # Create cross-validation strategy
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds, 
            shuffle=True, 
            random_state=self.config.random_state
        )
        
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Preprocess data
            X_train_processed = self.preprocessor.fit_transform(X_train_fold)
            X_val_processed = self.preprocessor.transform(X_val_fold)
            
            # Train model
            model = xgb.XGBClassifier(**param)
            model.fit(
                X_train_processed, y_train_fold,
                eval_set=[(X_val_processed, y_val_fold)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_val_processed)[:, 1]
            auc_score = roc_auc_score(y_val_fold, y_pred_proba)
            cv_scores.append(auc_score)
            
            logger.info(f"Fold {fold + 1} AUC: {auc_score:.4f}")
        
        mean_auc = np.mean(cv_scores)
        logger.info(f"Mean CV AUC: {mean_auc:.4f}")
        
        return mean_auc
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Run hyperparameter optimization with Optuna."""
        logger.info("Starting hyperparameter optimization...")
        
        # Create preprocessor
        self.preprocessor = self.create_preprocessor()
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
        )
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=self.config.optuna.n_trials,
            timeout=self.config.optuna.timeout
        )
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Best CV AUC: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Log to MLflow
        with mlflow.start_run(run_name="hyperparameter_optimization"):
            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_auc", best_score)
            mlflow.log_optuna_study(study)
        
        return best_params
    
    def train_final_model(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any]):
        """Train final model with best hyperparameters."""
        logger.info("Training final model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.config.validation_split,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Preprocess data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_val_processed = self.preprocessor.transform(X_val)
        
        # Create final model
        final_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'random_state': self.config.random_state,
            **params
        }
        
        self.model = xgb.XGBClassifier(**final_params)
        
        # Train with early stopping
        self.model.fit(
            X_train_processed, y_train,
            eval_set=[(X_val_processed, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate
        y_pred_proba = self.model.predict_proba(X_val_processed)[:, 1]
        y_pred = self.model.predict(X_val_processed)
        
        auc_score = roc_auc_score(y_val, y_pred_proba)
        classification_rep = classification_report(y_val, y_pred)
        conf_matrix = confusion_matrix(y_val, y_pred)
        
        logger.info(f"Final model AUC: {auc_score:.4f}")
        logger.info(f"Classification Report:\n{classification_rep}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
        # Log to MLflow
        with mlflow.start_run(run_name="final_model"):
            mlflow.log_params(final_params)
            mlflow.log_metric("validation_auc", auc_score)
            mlflow.xgboost.log_model(self.model, "model")
            mlflow.sklearn.log_model(self.preprocessor, "preprocessor")
            
            # Log metrics
            precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
            recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
            f1 = 2 * (precision * recall) / (precision + recall)
            
            mlflow.log_metrics({
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })
        
        return auc_score
    
    def save_model(self, model_path: str, preprocessor_path: str):
        """Save trained model and preprocessor."""
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model not trained yet")
        
        # Save model
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save preprocessor
        joblib.dump(self.preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")
        
        # Save metadata
        metadata = {
            'model_type': 'XGBoost',
            'feature_names': self.model.get_booster().feature_names,
            'model_params': self.model.get_params(),
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        with open(model_path.replace('.pkl', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

@hydra.main(version_base=None, config_path="../../config", config_name="training")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Setup logging
    setup_logging(cfg.logging)
    
    logger.info("Starting model training...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Initialize trainer
    trainer = LoanDefaultTrainer(cfg)
    
    # Load data
    X, y = trainer.load_data()
    
    # Optimize hyperparameters
    best_params = trainer.optimize_hyperparameters(X, y)
    
    # Train final model
    auc_score = trainer.train_final_model(X, y, best_params)
    
    # Save model
    model_dir = Path(cfg.model.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(
        model_path=model_dir / "model.pkl",
        preprocessor_path=model_dir / "preprocessor.pkl"
    )
    
    logger.info(f"Training completed successfully. Final AUC: {auc_score:.4f}")

if __name__ == "__main__":
    main()
