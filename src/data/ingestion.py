from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


class DataIngestion:
    def __init__(self, config):
        self.config = config

    def load_data(self, source: str | pd.DataFrame, source_type: str = "csv", validate: bool = True) -> pd.DataFrame:
        del validate
        if isinstance(source, pd.DataFrame):
            return source.copy()

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Data source not found: {source}")

        if source_type == "csv":
            df = pd.read_csv(path)
        elif source_type == "json":
            df = pd.read_json(path)
        elif source_type == "parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        logger.info("Loaded %s rows from %s", len(df), source)
        return df

    def get_data_quality_report(self, df: pd.DataFrame) -> dict:
        return {
            "total_records": int(len(df)),
            "total_features": int(len(df.columns)),
            "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
            "duplicate_records": int(df.duplicated().sum()),
        }
