# Model Development

- Train: `python src/models/train.py`
- Evaluate: `python src/models/evaluate.py --model-path models/production/model.pkl`
- Optimize report: `python scripts/optimize_model.py --model-path models/production/model.pkl --preprocessor-path models/production/preprocessor.pkl --test-data <X.csv> --test-labels <y.csv> --output-dir models/optimized`
