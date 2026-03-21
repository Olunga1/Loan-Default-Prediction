from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import redis
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest

from src.api.middleware.auth import authenticate_request
from src.api.middleware.logging import RequestLoggingMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.schemas import (
    BatchPredictionRequest,
    ExplanationResponse,
    LoanApplicationRequest,
    PredictionResponse,
)
from src.monitoring.metrics import (
    active_connections_gauge,
    prediction_counter,
    prediction_latency,
    registry,
)
from src.utils.config import get_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

config = get_config()
model = None
preprocessor = None
redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor, redis_client

    model_path = Path(config.model.path)
    preproc_path = Path(config.model.preprocessor_path)

    if model_path.exists() and preproc_path.exists():
        model = joblib.load(model_path)
        preprocessor = joblib.load(preproc_path)
        logger.info("Loaded model artifacts")
    else:
        logger.warning("Model artifacts not found at %s and %s", model_path, preproc_path)

    try:
        redis_client = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            decode_responses=True,
        )
        redis_client.ping()
    except Exception as exc:
        logger.warning("Redis unavailable: %s", exc)
        redis_client = None

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)

    yield

    if redis_client is not None:
        redis_client.close()


app = FastAPI(
    title="Loan Default Prediction API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)


@app.middleware("http")
async def track_active_connections(request, call_next):
    active_connections_gauge.inc()
    try:
        return await call_next(request)
    finally:
        active_connections_gauge.dec()


@app.get("/health")
async def health_check():
    redis_ok = False
    if redis_client is not None:
        try:
            redis_ok = bool(redis_client.ping())
        except Exception:
            redis_ok = False

    healthy = model is not None and preprocessor is not None
    payload = {
        "status": "healthy" if healthy else "degraded",
        "timestamp": time.time(),
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "redis_connected": redis_ok,
    }
    return JSONResponse(status_code=200, content=payload)


@app.get("/metrics")
async def metrics():
    if not config.monitoring.enabled:
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    return Response(content=generate_latest(registry), media_type="text/plain; version=0.0.4")


def _risk_category(default_probability: float) -> str:
    if default_probability < 0.2:
        return "Low"
    if default_probability < 0.5:
        return "Medium"
    return "High"


def _ensure_model_ready() -> None:
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: LoanApplicationRequest,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(authenticate_request),
):
    del authenticated
    _ensure_model_ready()
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())

    try:
        cache_key = f"prediction:{hash(request.model_dump_json())}"
        if redis_client is not None:
            cached = redis_client.get(cache_key)
            if cached:
                cached_resp = PredictionResponse.model_validate_json(cached)
                prediction_counter.labels(
                    risk_category=cached_resp.risk_category,
                    cache_hit="true",
                    model_version=cached_resp.model_version,
                ).inc()
                return cached_resp

        input_df = pd.DataFrame([request.model_dump()])
        processed = preprocessor.transform(input_df)
        proba = float(model.predict_proba(processed)[0][1])
        risk = _risk_category(proba)

        response = PredictionResponse(
            loan_id=request.loan_id,
            default_probability=proba,
            risk_category=risk,
            prediction_timestamp=time.time(),
            model_version=config.model.version,
        )

        if redis_client is not None:
            redis_client.setex(cache_key, config.redis.ttl, response.model_dump_json())

        prediction_counter.labels(
            risk_category=risk,
            cache_hit="false",
            model_version=config.model.version,
        ).inc()
        prediction_latency.observe(time.perf_counter() - start_time)

        background_tasks.add_task(log_prediction_async, request_id, request.model_dump(), response.model_dump())
        return response
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(
    request: BatchPredictionRequest,
    authenticated: bool = Depends(authenticate_request),
):
    del authenticated
    _ensure_model_ready()

    if len(request.applications) > config.batch.max_size:
        raise HTTPException(status_code=400, detail=f"Batch size exceeds maximum of {config.batch.max_size}")

    df = pd.DataFrame([app.model_dump() for app in request.applications])
    processed = preprocessor.transform(df)
    probas = model.predict_proba(processed)

    out = []
    for app_data, pred in zip(request.applications, probas):
        p_default = float(pred[1])
        out.append(
            PredictionResponse(
                loan_id=app_data.loan_id,
                default_probability=p_default,
                risk_category=_risk_category(p_default),
                prediction_timestamp=time.time(),
                model_version=config.model.version,
            )
        )
    return out


@app.post("/explain", response_model=ExplanationResponse)
async def explain(
    request: LoanApplicationRequest,
    authenticated: bool = Depends(authenticate_request),
):
    del authenticated
    _ensure_model_ready()

    try:
        import shap

        input_df = pd.DataFrame([request.model_dump()])
        processed = preprocessor.transform(input_df)
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(processed)
        values = shap_values[0] if hasattr(shap_values, "__len__") else shap_values

        contributions = {
            name: float(val)
            for name, val in zip(feature_names, values)
        }

        return ExplanationResponse(
            loan_id=request.loan_id,
            feature_contributions=dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)),
            base_value=float(getattr(explainer, "expected_value", 0.0)),
            explanation_timestamp=time.time(),
            model_version=config.model.version,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}") from exc


async def log_prediction_async(request_id: str, input_data: dict, output_data: dict) -> None:
    try:
        with mlflow.start_run(run_name=f"prediction_{request_id}"):
            mlflow.log_params(input_data)
            mlflow.log_metric("default_probability", float(output_data.get("default_probability", 0.0)))
    except Exception as exc:
        logger.warning("Failed to log prediction %s: %s", request_id, exc)


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
