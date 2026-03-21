from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class RiskCategory(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class EducationLevel(str, Enum):
    HIGH_SCHOOL = "High School"
    BACHELORS = "Bachelor's"
    MASTERS = "Master's"
    PHD = "PhD"


class EmploymentType(str, Enum):
    FULL_TIME = "Full-time"
    PART_TIME = "Part-time"
    SELF_EMPLOYED = "Self-employed"
    UNEMPLOYED = "Unemployed"


class MaritalStatus(str, Enum):
    SINGLE = "Single"
    MARRIED = "Married"
    DIVORCED = "Divorced"


class LoanPurpose(str, Enum):
    HOME = "Home"
    AUTO = "Auto"
    EDUCATION = "Education"
    BUSINESS = "Business"
    OTHER = "Other"


class YesNo(str, Enum):
    YES = "Yes"
    NO = "No"


class LoanApplicationRequest(BaseModel):
    loan_id: str = Field(..., min_length=1, max_length=50)
    age: int = Field(..., ge=18, le=100)
    income: int = Field(..., ge=15000, le=10000000)
    loan_amount: int = Field(..., ge=1000, le=10000000)
    credit_score: int = Field(..., ge=300, le=850)
    months_employed: int = Field(..., ge=0, le=600)
    num_credit_lines: int = Field(..., ge=1, le=20)
    interest_rate: float = Field(..., ge=0.0, le=50.0)
    loan_term: int = Field(..., ge=12, le=360)
    dti_ratio: float = Field(..., ge=0.0, le=0.9)
    education: EducationLevel
    employment_type: EmploymentType
    marital_status: MaritalStatus
    has_mortgage: YesNo
    has_dependents: YesNo
    loan_purpose: LoanPurpose
    has_cosigner: YesNo

    @field_validator("dti_ratio")
    @classmethod
    def validate_dti(cls, value: float) -> float:
        if value > 0.8:
            raise ValueError("Debt-to-income ratio cannot exceed 80%")
        return value

    @field_validator("loan_amount")
    @classmethod
    def validate_loan_amount(cls, value: int, info):
        income = info.data.get("income") if info and info.data else None
        if income and value > income * 10:
            raise ValueError("Loan amount cannot exceed 10x annual income")
        return value


class PredictionResponse(BaseModel):
    loan_id: str
    default_probability: float = Field(..., ge=0.0, le=1.0)
    risk_category: RiskCategory
    prediction_timestamp: float
    model_version: str


class ExplanationResponse(BaseModel):
    loan_id: str
    feature_contributions: dict[str, float]
    base_value: float
    explanation_timestamp: float
    model_version: str


class BatchPredictionRequest(BaseModel):
    applications: list[LoanApplicationRequest] = Field(..., min_length=1, max_length=1000)


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    model_loaded: bool
    preprocessor_loaded: bool
    redis_connected: bool
    uptime: float | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    timestamp: float
    request_id: str | None = None


class ModelMetrics(BaseModel):
    auc_score: float = Field(..., ge=0.0, le=1.0)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    confusion_matrix: list[list[int]]
    model_version: str
    evaluation_date: datetime


class DataDriftReport(BaseModel):
    drift_detected: bool
    drift_score: float = Field(..., ge=0.0, le=1.0)
    feature_drift: dict[str, float]
    report_timestamp: datetime
    baseline_dataset: str
    current_dataset: str
