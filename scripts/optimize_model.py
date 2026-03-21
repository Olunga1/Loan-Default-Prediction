#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline model optimization report")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--preprocessor-path", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--test-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    preprocessor = joblib.load(args.preprocessor_path)

    X = pd.read_csv(args.test_data)
    y = pd.read_csv(args.test_labels).squeeze()

    Xp = preprocessor.transform(X)

    start = time.perf_counter()
    p = model.predict_proba(Xp)[:, 1]
    duration = time.perf_counter() - start

    report = {
        "auc": float(roc_auc_score(y, p)),
        "rows": int(len(X)),
        "total_inference_seconds": float(duration),
        "avg_inference_ms": float((duration / max(len(X), 1)) * 1000),
        "recommendations": [
            "Export to ONNX in a follow-up step for runtime portability.",
            "Enable batch inference for high-throughput workloads.",
            "Cache repeated feature vectors with short TTL.",
        ],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "optimization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
