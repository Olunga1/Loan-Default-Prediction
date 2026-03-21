# Architecture Overview

Minimal baseline architecture:
- Next.js frontend (`frontend`) calls FastAPI backend (`src/api`).
- FastAPI serves `/predict`, `/predict/batch`, `/explain`, `/health`, `/metrics`.
- Model artifacts are loaded from `models/production`.
- Optional Redis caching and MLflow logging are enabled via config.
