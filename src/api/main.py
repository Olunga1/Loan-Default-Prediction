"""
FastAPI inference service for loan default prediction.
Provides real-time predictions with explainability and monitoring.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import time
import uuid
from typing import Dict, List, Optional
import asyncio
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
import mlflow
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import redis
from pydantic import BaseModel, Field

from .schemas import LoanApplicationRequest, PredictionResponse, ExplanationResponse
from .middleware.auth import authenticate_request
from .middleware.logging import RequestLoggingMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from ..utils.config import get_config
from ..monitoring.metrics import (
    prediction_counter, 
    prediction_latency, 
    model_accuracy_gauge,
    active_connections_gauge
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
preprocessor = None
redis_client = None
config = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    global model, preprocessor, redis_client, config
    
    logger.info("Starting up inference service...")
    
    # Load configuration
    config = get_config()
    
    # Load model and preprocessor
    try:
        model = joblib.load(config.model.path)
        preprocessor = joblib.load(config.preprocessor.path)
        logger.info("Model and preprocessor loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Initialize Redis
    try:
        redis_client = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            decode_responses=True
        )
        redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None
    
    # Setup MLflow
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    
    logger.info("Service startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down inference service...")
    if redis_client:
        redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="Loan Default Prediction API",
    description="Production-grade API for loan default risk assessment with explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
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

# Metrics endpoints
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if not config.monitoring.enabled:
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    return generate_latest()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "redis_connected": redis_client is not None and redis_client.ping()
    }
    
    # Check if all components are healthy
    if all(health_status.values()):
        return JSONResponse(status_code=200, content=health_status)
    else:
        return JSONResponse(status_code=503, content=health_status)

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: LoanApplicationRequest,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(authenticate_request)
):
    """
    Predict loan default probability.
    
    Args:
        request: Loan application data
        background_tasks: Background tasks for async operations
        authenticated: Authentication status from middleware
        
    Returns:
        Prediction response with probability and risk category
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Log prediction request
        logger.info(f"Prediction request {request_id}: {request.loan_id}")
        
        # Check cache first
        cache_key = f"prediction:{hash(str(request.dict()))}"
        if redis_client:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for request {request_id}")
                prediction_counter.labels(cache_hit="true").inc()
                return PredictionResponse.parse_raw(cached_result)
        
        # Convert to DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Preprocess
        processed_data = preprocessor.transform(input_data)
        
        # Make prediction
        prediction_proba = model.predict_proba(processed_data)[0]
        default_probability = float(prediction_proba[1])
        
        # Determine risk category
        if default_probability < 0.2:
            risk_category = "Low"
        elif default_probability < 0.5:
            risk_category = "Medium"
        else:
            risk_category = "High"
        
        # Create response
        response = PredictionResponse(
            loan_id=request.loan_id,
            default_probability=default_probability,
            risk_category=risk_category,
            prediction_timestamp=time.time(),
            model_version=config.model.version
        )
        
        # Cache result
        if redis_client:
            redis_client.setex(
                cache_key, 
                config.redis.ttl, 
                response.json()
            )
        
        # Log prediction
        prediction_counter.labels(
            risk_category=risk_category,
            cache_hit="false"
        ).inc()
        
        # Background task for monitoring
        background_tasks.add_task(
            log_prediction_async,
            request_id,
            request.dict(),
            response.dict()
        )
        
        # Record latency
        latency = time.time() - start_time
        prediction_latency.observe(latency)
        
        logger.info(f"Prediction {request_id} completed in {latency:.3f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed for request {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/explain", response_model=ExplanationResponse)
async def explain(
    request: LoanApplicationRequest,
    authenticated: bool = Depends(authenticate_request)
):
    """
    Generate model explanation using SHAP values.
    
    Args:
        request: Loan application data
        authenticated: Authentication status
        
    Returns:
        Explanation response with feature contributions
    """
    try:
        import shap
        
        # Convert to DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Preprocess
        processed_data = preprocessor.transform(input_data)
        
        # Get feature names
        feature_names = preprocessor.get_feature_names_out()
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(processed_data)
        
        # Get feature contributions
        contributions = {}
        for i, (name, value) in enumerate(zip(feature_names, shap_values[0])):
            contributions[name] = float(value)
        
        # Sort by absolute contribution
        sorted_contributions = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        
        # Create explanation
        explanation = ExplanationResponse(
            loan_id=request.loan_id,
            feature_contributions=sorted_contributions,
            base_value=float(explainer.expected_value),
            explanation_timestamp=time.time(),
            model_version=config.model.version
        )
        
        return explanation
        
    except Exception as e:
        logger.error(f"Explanation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(
    requests: List[LoanApplicationRequest],
    authenticated: bool = Depends(authenticate_request)
):
    """
    Batch prediction endpoint for multiple loan applications.
    
    Args:
        requests: List of loan applications
        authenticated: Authentication status
        
    Returns:
        List of prediction responses
    """
    try:
        if len(requests) > config.batch.max_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Batch size exceeds maximum of {config.batch.max_size}"
            )
        
        # Convert to DataFrame
        input_data = pd.DataFrame([req.dict() for req in requests])
        
        # Preprocess
        processed_data = preprocessor.transform(input_data)
        
        # Make predictions
        prediction_probas = model.predict_proba(processed_data)
        
        # Create responses
        responses = []
        for i, request in enumerate(requests):
            default_probability = float(prediction_probas[i, 1])
            
            if default_probability < 0.2:
                risk_category = "Low"
            elif default_probability < 0.5:
                risk_category = "Medium"
            else:
                risk_category = "High"
            
            response = PredictionResponse(
                loan_id=request.loan_id,
                default_probability=default_probability,
                risk_category=risk_category,
                prediction_timestamp=time.time(),
                model_version=config.model.version
            )
            responses.append(response)
        
        logger.info(f"Batch prediction completed for {len(requests)} applications")
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

async def log_prediction_async(request_id: str, input_data: Dict, output_data: Dict):
    """Background task to log predictions for monitoring."""
    try:
        # Log to MLflow
        with mlflow.start_run(run_name=f"prediction_{request_id}"):
            mlflow.log_params(input_data)
            mlflow.log_metrics(output_data)
        
        # Log to database or monitoring system
        # This would integrate with your monitoring infrastructure
        
    except Exception as e:
        logger.error(f"Failed to log prediction {request_id}: {str(e)}")

# Update active connections
@app.middleware("http")
async def update_active_connections(request, call_next):
    active_connections_gauge.inc()
    try:
        response = await call_next(request)
        return response
    finally:
        active_connections_gauge.dec()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
