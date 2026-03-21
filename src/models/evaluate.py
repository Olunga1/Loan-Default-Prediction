from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--preprocessor-path", default="models/production/preprocessor.pkl")
    parser.add_argument("--data-path", default="Loan_default.csv")
    parser.add_argument("--target-col", default="Default")
    parser.add_argument("--output", default="models/production/eval_metrics.json")
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    preprocessor = joblib.load(args.preprocessor_path)

    df = pd.read_csv(args.data_path)
    target_col = args.target_col if args.target_col in df.columns else args.target_col.lower()
    y = df[target_col]
    feature_cols = [c for c in df.columns if c not in {target_col, "LoanID", "loan_id"}]
    X = df[feature_cols]

    Xp = preprocessor.transform(X)
    y_proba = model.predict_proba(Xp)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y, y_proba)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
