from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CorsSettings(BaseSettings):
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


class ModelSettings(BaseSettings):
    path: str = "models/production/model.pkl"
    preprocessor_path: str = "models/production/preprocessor.pkl"
    version: str = "0.1.0"


class RedisSettings(BaseSettings):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    ttl: int = 300


class MlflowSettings(BaseSettings):
    tracking_uri: str = "http://localhost:5000"


class MonitoringSettings(BaseSettings):
    enabled: bool = True


class BatchSettings(BaseSettings):
    max_size: int = 100


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    cors: CorsSettings = CorsSettings()
    model: ModelSettings = ModelSettings()
    redis: RedisSettings = RedisSettings()
    mlflow: MlflowSettings = MlflowSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    batch: BatchSettings = BatchSettings()


def get_config() -> Settings:
    return Settings()


def setup_logging(logging_cfg) -> None:
    import logging

    level_name = str(getattr(logging_cfg, "level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = str(getattr(logging_cfg, "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.basicConfig(level=level, format=fmt)
