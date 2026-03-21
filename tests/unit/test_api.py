"""
Unit tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json
import pandas as pd

from src.api.main import app
from src.api.schemas import LoanApplicationRequest

client = TestClient(app)

class TestPredictionEndpoint:
    """Test cases for prediction endpoint."""
    
    def test_predict_valid_request(self):
        """Test prediction with valid request."""
        request_data = {
            "loan_id": "TEST001",
            "age": 35,
            "income": 75000,
            "loan_amount": 250000,
            "credit_score": 720,
            "months_employed": 60,
            "num_credit_lines": 3,
            "interest_rate": 4.5,
            "loan_term": 360,
            "dti_ratio": 0.35,
            "education": "Bachelor's",
            "employment_type": "Full-time",
            "marital_status": "Married",
            "has_mortgage": "Yes",
            "has_dependents": "Yes",
            "loan_purpose": "Home",
            "has_cosigner": "No"
        }
        
        with patch('src.api.main.model') as mock_model, \
             patch('src.api.main.preprocessor') as mock_preprocessor:
            
            # Mock model prediction
            mock_model.predict_proba.return_value = [[0.8, 0.2]]
            mock_preprocessor.transform.return_value = [[1, 2, 3, 4, 5]]
            
            response = client.post("/predict", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["loan_id"] == "TEST001"
            assert data["default_probability"] == 0.2
            assert data["risk_category"] in ["Low", "Medium", "High"]
    
    def test_predict_invalid_request(self):
        """Test prediction with invalid request."""
        invalid_data = {
            "loan_id": "TEST001",
            "age": 17,  # Invalid age
            "income": 75000,
            # Missing required fields
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422
    
    def test_predict_high_risk_application(self):
        """Test prediction for high-risk application."""
        request_data = {
            "loan_id": "HIGH_RISK_001",
            "age": 22,
            "income": 20000,
            "loan_amount": 500000,
            "credit_score": 400,
            "months_employed": 6,
            "num_credit_lines": 8,
            "interest_rate": 15.0,
            "loan_term": 360,
            "dti_ratio": 0.85,
            "education": "High School",
            "employment_type": "Unemployed",
            "marital_status": "Single",
            "has_mortgage": "No",
            "has_dependents": "No",
            "loan_purpose": "Other",
            "has_cosigner": "No"
        }
        
        with patch('src.api.main.model') as mock_model, \
             patch('src.api.main.preprocessor') as mock_preprocessor:
            
            mock_model.predict_proba.return_value = [[0.3, 0.7]]
            mock_preprocessor.transform.return_value = [[1, 2, 3, 4, 5]]
            
            response = client.post("/predict", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["risk_category"] == "High"
            assert data["default_probability"] > 0.5

class TestExplanationEndpoint:
    """Test cases for explanation endpoint."""
    
    def test_explain_valid_request(self):
        """Test explanation with valid request."""
        request_data = {
            "loan_id": "EXPLAIN001",
            "age": 35,
            "income": 75000,
            "loan_amount": 250000,
            "credit_score": 720,
            "months_employed": 60,
            "num_credit_lines": 3,
            "interest_rate": 4.5,
            "loan_term": 360,
            "dti_ratio": 0.35,
            "education": "Bachelor's",
            "employment_type": "Full-time",
            "marital_status": "Married",
            "has_mortgage": "Yes",
            "has_dependents": "Yes",
            "loan_purpose": "Home",
            "has_cosigner": "No"
        }
        
        with patch('src.api.main.model') as mock_model, \
             patch('src.api.main.preprocessor') as mock_preprocessor, \
             patch('shap.TreeExplainer') as mock_shap:
            
            # Mock SHAP explanation
            mock_shap.return_value.shap_values.return_value = [[0.1, -0.2, 0.05, -0.1, 0.15]]
            mock_shap.return_value.expected_value = 0.116
            mock_preprocessor.transform.return_value = [[1, 2, 3, 4, 5]]
            mock_preprocessor.get_feature_names_out.return_value = [
                "CreditScore", "Income", "LoanAmount", "DTIRatio", "Age"
            ]
            
            response = client.post("/explain", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["loan_id"] == "EXPLAIN001"
            assert "feature_contributions" in data
            assert "base_value" in data

class TestBatchPrediction:
    """Test cases for batch prediction endpoint."""
    
    def test_batch_prediction_valid(self):
        """Test batch prediction with valid requests."""
        batch_data = {
            "applications": [
                {
                    "loan_id": "BATCH001",
                    "age": 35,
                    "income": 75000,
                    "loan_amount": 250000,
                    "credit_score": 720,
                    "months_employed": 60,
                    "num_credit_lines": 3,
                    "interest_rate": 4.5,
                    "loan_term": 360,
                    "dti_ratio": 0.35,
                    "education": "Bachelor's",
                    "employment_type": "Full-time",
                    "marital_status": "Married",
                    "has_mortgage": "Yes",
                    "has_dependents": "Yes",
                    "loan_purpose": "Home",
                    "has_cosigner": "No"
                },
                {
                    "loan_id": "BATCH002",
                    "age": 28,
                    "income": 50000,
                    "loan_amount": 150000,
                    "credit_score": 680,
                    "months_employed": 36,
                    "num_credit_lines": 2,
                    "interest_rate": 5.5,
                    "loan_term": 240,
                    "dti_ratio": 0.42,
                    "education": "Master's",
                    "employment_type": "Full-time",
                    "marital_status": "Single",
                    "has_mortgage": "No",
                    "has_dependents": "No",
                    "loan_purpose": "Auto",
                    "has_cosigner": "Yes"
                }
            ]
        }
        
        with patch('src.api.main.model') as mock_model, \
             patch('src.api.main.preprocessor') as mock_preprocessor:
            
            mock_model.predict_proba.return_value = [[0.8, 0.2], [0.9, 0.1]]
            mock_preprocessor.transform.return_value = [[1, 2, 3], [4, 5, 6]]
            
            response = client.post("/predict/batch", json=batch_data)
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["loan_id"] == "BATCH001"
            assert data[1]["loan_id"] == "BATCH002"
    
    def test_batch_prediction_too_large(self):
        """Test batch prediction with too many applications."""
        # Create batch with 1001 applications (exceeds limit)
        applications = []
        for i in range(1001):
            applications.append({
                "loan_id": f"BIG_BATCH_{i:03d}",
                "age": 35,
                "income": 75000,
                "loan_amount": 250000,
                "credit_score": 720,
                "months_employed": 60,
                "num_credit_lines": 3,
                "interest_rate": 4.5,
                "loan_term": 360,
                "dti_ratio": 0.35,
                "education": "Bachelor's",
                "employment_type": "Full-time",
                "marital_status": "Married",
                "has_mortgage": "Yes",
                "has_dependents": "Yes",
                "loan_purpose": "Home",
                "has_cosigner": "No"
            })
        
        batch_data = {"applications": applications}
        
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 400

class TestHealthEndpoint:
    """Test cases for health check endpoint."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data
        assert "preprocessor_loaded" in data
        assert "redis_connected" in data

class TestAuthentication:
    """Test cases for authentication middleware."""
    
    def test_unauthorized_request(self):
        """Test request without authentication."""
        # This test would need to be implemented based on your auth strategy
        # For now, we'll assume the endpoint requires auth
        request_data = {
            "loan_id": "AUTH_TEST",
            "age": 35,
            "income": 75000,
            "loan_amount": 250000,
            "credit_score": 720,
            "months_employed": 60,
            "num_credit_lines": 3,
            "interest_rate": 4.5,
            "loan_term": 360,
            "dti_ratio": 0.35,
            "education": "Bachelor's",
            "employment_type": "Full-time",
            "marital_status": "Married",
            "has_mortgage": "Yes",
            "has_dependents": "Yes",
            "loan_purpose": "Home",
            "has_cosigner": "No"
        }
        
        # Test without auth header (implementation depends on your auth middleware)
        response = client.post("/predict", json=request_data)
        # This might return 200 if auth is not enforced in tests
        # Adjust based on your auth implementation

class TestRateLimiting:
    """Test cases for rate limiting."""
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        request_data = {
            "loan_id": "RATE_LIMIT_TEST",
            "age": 35,
            "income": 75000,
            "loan_amount": 250000,
            "credit_score": 720,
            "months_employed": 60,
            "num_credit_lines": 3,
            "interest_rate": 4.5,
            "loan_term": 360,
            "dti_ratio": 0.35,
            "education": "Bachelor's",
            "employment_type": "Full-time",
            "marital_status": "Married",
            "has_mortgage": "Yes",
            "has_dependents": "Yes",
            "loan_purpose": "Home",
            "has_cosigner": "No"
        }
        
        # This test would need to be implemented based on your rate limiting strategy
        # You might need to make multiple requests quickly and check for 429 status
        pass

if __name__ == "__main__":
    pytest.main([__file__])
