"""
Prometheus metrics for monitoring the loan default prediction service.
"""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time
import logging

logger = logging.getLogger(__name__)

# Create a custom registry
registry = CollectorRegistry()

# Prediction metrics
prediction_counter = Counter(
    'loan_predictions_total',
    'Total number of loan predictions made',
    ['risk_category', 'cache_hit', 'model_version'],
    registry=registry
)

prediction_latency = Histogram(
    'loan_prediction_duration_seconds',
    'Time spent processing loan predictions',
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0],
    registry=registry
)

model_accuracy_gauge = Gauge(
    'loan_model_accuracy',
    'Current model accuracy score',
    ['model_version'],
    registry=registry
)

active_connections_gauge = Gauge(
    'loan_active_connections',
    'Number of active connections',
    registry=registry
)

# Data quality metrics
data_quality_score = Gauge(
    'loan_data_quality_score',
    'Data quality score (0-1)',
    ['data_source'],
    registry=registry
)

missing_values_counter = Counter(
    'loan_missing_values_total',
    'Total number of missing values detected',
    ['feature'],
    registry=registry
)

# Model performance metrics
auc_score_gauge = Gauge(
    'loan_model_auc_score',
    'Model ROC AUC score',
    ['model_version', 'dataset'],
    registry=registry
)

drift_detection_score = Gauge(
    'loan_drift_detection_score',
    'Data drift detection score',
    ['feature'],
    registry=registry
)

# Business metrics
high_risk_applications_counter = Counter(
    'loan_high_risk_applications_total',
    'Total number of high-risk applications',
    ['model_version'],
    registry=registry
)

loan_amount_sum = Counter(
    'loan_amount_sum_total',
    'Total sum of loan amounts processed',
    ['risk_category'],
    registry=registry
)

# System metrics
cache_hit_ratio = Gauge(
    'loan_cache_hit_ratio',
    'Cache hit ratio',
    ['cache_type'],
    registry=registry
)

error_counter = Counter(
    'loan_errors_total',
    'Total number of errors',
    ['error_type', 'endpoint'],
    registry=registry
)

class MetricsCollector:
    """Centralized metrics collection and reporting."""
    
    def __init__(self):
        self.start_time = time.time()
    
    def record_prediction(self, risk_category: str, cache_hit: bool, model_version: str):
        """Record a prediction event."""
        prediction_counter.labels(
            risk_category=risk_category,
            cache_hit=str(cache_hit).lower(),
            model_version=model_version
        ).inc()
    
    def record_prediction_latency(self, duration: float):
        """Record prediction processing time."""
        prediction_latency.observe(duration)
    
    def update_model_accuracy(self, accuracy: float, model_version: str):
        """Update model accuracy metric."""
        model_accuracy_gauge.labels(model_version=model_version).set(accuracy)
    
    def update_auc_score(self, auc: float, model_version: str, dataset: str):
        """Update AUC score metric."""
        auc_score_gauge.labels(model_version=model_version, dataset=dataset).set(auc)
    
    def record_high_risk_application(self, model_version: str):
        """Record a high-risk application."""
        high_risk_applications_counter.labels(model_version=model_version).inc()
    
    def record_loan_amount(self, amount: float, risk_category: str):
        """Record loan amount processed."""
        loan_amount_sum.labels(risk_category=risk_category).inc(amount)
    
    def record_error(self, error_type: str, endpoint: str):
        """Record an error event."""
        error_counter.labels(error_type=error_type, endpoint=endpoint).inc()
    
    def update_data_quality(self, score: float, data_source: str):
        """Update data quality score."""
        data_quality_score.labels(data_source=data_source).set(score)
    
    def record_missing_values(self, feature: str, count: int):
        """Record missing values count."""
        missing_values_counter.labels(feature=feature).inc(count)
    
    def update_drift_score(self, score: float, feature: str):
        """Update drift detection score."""
        drift_detection_score.labels(feature=feature).set(score)
    
    def update_cache_hit_ratio(self, ratio: float, cache_type: str):
        """Update cache hit ratio."""
        cache_hit_ratio.labels(cache_type=cache_type).set(ratio)
    
    def get_uptime(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self.start_time

# Global metrics collector instance
metrics = MetricsCollector()

def init_metrics():
    """Initialize metrics collection."""
    logger.info("Metrics collection initialized")
    
    # Set initial values
    active_connections_gauge.set(0)
    cache_hit_ratio.labels(cache_type='prediction').set(0.0)
    
    logger.info("Initial metrics set")

class PredictionTimer:
    """Context manager for timing predictions."""
    
    def __init__(self, record_func):
        self.record_func = record_func
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.record_func(duration)

def time_prediction(func):
    """Decorator to time prediction functions."""
    def wrapper(*args, **kwargs):
        with PredictionTimer(metrics.record_prediction_latency):
            return func(*args, **kwargs)
    return wrapper
