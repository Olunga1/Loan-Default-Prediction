"""
Pydantic schemas for API request/response validation.
Defines data models for loan applications and predictions.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

class RiskCategory(str, Enum):
    """Risk category enumeration."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class EducationLevel(str, Enum):
    """Education level enumeration."""
    HIGH_SCHOOL = "High School"
    BACHELORS = "Bachelor's"
    MASTERS = "Master's"
    PHD = "PhD"

class EmploymentType(str, Enum):
    """Employment type enumeration."""
    FULL_TIME = "Full-time"
    PART_TIME = "Part-time"
    SELF_EMPLOYED = "Self-employed"
    UNEMPLOYED = "Unemployed"

class MaritalStatus(str, Enum):
    """Marital status enumeration."""
    SINGLE = "Single"
    MARRIED = "Married"
    DIVORCED = "Divorced"

class LoanPurpose(str, Enum):
    """Loan purpose enumeration."""
    HOME = "Home"
    AUTO = "Auto"
    EDUCATION = "Education"
    BUSINESS = "Business"
    OTHER = "Other"

class YesNo(str, Enum):
    """Yes/No enumeration."""
    YES = "Yes"
    NO = "No"

class LoanApplicationRequest(BaseModel):
    """Request model for loan application prediction."""
    
    loan_id: str = Field(..., min_length=1, max_length=50, description="Unique loan identifier")
    age: int = Field(..., ge=18, le=100, description="Borrower age in years")
    income: int = Field(..., ge=15000, le=10000000, description="Annual income in USD")
    loan_amount: int = Field(..., ge=1000, le=10000000, description="Loan amount in USD")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    months_employed: int = Field(..., ge=0, le=600, description="Months employed")
    num_credit_lines: int = Field(..., ge=1, le=20, description="Number of credit lines")
    interest_rate: float = Field(..., ge=0.0, le=50.0, description="Interest rate percentage")
    loan_term: int = Field(..., ge=12, le=360, description="Loan term in months")
    dti_ratio: float = Field(..., ge=0.0, le=0.9, description="Debt-to-income ratio")
    education: EducationLevel = Field(..., description="Highest education level")
    employment_type: EmploymentType = Field(..., description="Current employment type")
    marital_status: MaritalStatus = Field(..., description="Marital status")
    has_mortgage: YesNo = Field(..., description="Has existing mortgage")
    has_dependents: YesNo = Field(..., description="Has dependents")
    loan_purpose: LoanPurpose = Field(..., description="Purpose of loan")
    has_cosigner: YesNo = Field(..., description="Has co-signer")
    
    @validator('income')
    def validate_income(cls, v):
        if v < 15000:
            raise ValueError('Income must be at least $15,000 for loan consideration')
        return v
    
    @validator('dti_ratio')
    def validate_dti(cls, v):
        if v > 0.8:
            raise ValueError('Debt-to-income ratio cannot exceed 80%')
        return v
    
    @validator('loan_amount')
    def validate_loan_amount(cls, v, values):
        if 'income' in values and v > values['income'] * 10:
            raise ValueError('Loan amount cannot exceed 10x annual income')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "loan_id": "LN123456789",
                "age": 35,
                "income": 85000,
                "loan_amount": 250000,
                "credit_score": 720,
                "months_employed": 72,
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
        }

class PredictionResponse(BaseModel):
    """Response model for loan default prediction."""
    
    loan_id: str = Field(..., description="Loan identifier")
    default_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of default")
    risk_category: RiskCategory = Field(..., description="Risk category")
    prediction_timestamp: float = Field(..., description="Unix timestamp of prediction")
    model_version: str = Field(..., description="Model version used")
    
    @validator('default_probability')
    def validate_probability(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Probability must be between 0 and 1')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "loan_id": "LN123456789",
                "default_probability": 0.156,
                "risk_category": "Low",
                "prediction_timestamp": 1703078400.0,
                "model_version": "1.0.0"
            }
        }

class ExplanationResponse(BaseModel):
    """Response model for model explanation."""
    
    loan_id: str = Field(..., description="Loan identifier")
    feature_contributions: Dict[str, float] = Field(..., description="Feature SHAP values")
    base_value: float = Field(..., description="Base prediction value")
    explanation_timestamp: float = Field(..., description="Unix timestamp of explanation")
    model_version: str = Field(..., description="Model version used")
    
    class Config:
        schema_extra = {
            "example": {
                "loan_id": "LN123456789",
                "feature_contributions": {
                    "CreditScore": -0.234,
                    "Income": -0.156,
                    "LoanAmount": 0.089,
                    "DTIRatio": 0.067,
                    "Age": -0.045
                },
                "base_value": 0.116,
                "explanation_timestamp": 1703078400.0,
                "model_version": "1.0.0"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    applications: List[LoanApplicationRequest] = Field(..., min_items=1, max_items=1000)
    
    @validator('applications')
    def validate_applications(cls, v):
        if len(v) > 1000:
            raise ValueError('Maximum 1000 applications per batch')
        return v

class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    timestamp: float = Field(..., description="Unix timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    preprocessor_loaded: bool = Field(..., description="Whether preprocessor is loaded")
    redis_connected: bool = Field(..., description="Whether Redis is connected")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")

class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: float = Field(..., description="Unix timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")

class ModelMetrics(BaseModel):
    """Model performance metrics."""
    
    auc_score: float = Field(..., ge=0.0, le=1.0, description="ROC AUC score")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Accuracy")
    precision: float = Field(..., ge=0.0, le=1.0, description="Precision")
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1 score")
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix")
    model_version: str = Field(..., description="Model version")
    evaluation_date: datetime = Field(..., description="Evaluation date")

class DataDriftReport(BaseModel):
    """Data drift detection report."""
    
    drift_detected: bool = Field(..., description="Whether drift was detected")
    drift_score: float = Field(..., ge=0.0, le=1.0, description="Drift score")
    feature_drift: Dict[str, float] = Field(..., description="Feature-wise drift scores")
    report_timestamp: datetime = Field(..., description="Report generation timestamp")
    baseline_dataset: str = Field(..., description="Baseline dataset used")
    current_dataset: str = Field(..., description="Current dataset analyzed")
