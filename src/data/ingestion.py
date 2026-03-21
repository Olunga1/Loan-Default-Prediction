"""
Data ingestion and validation for loan default prediction.
Handles data loading from various sources with robust validation.
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pydantic import BaseModel, validator, Field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LoanApplication(BaseModel):
    """Pydantic model for loan application data validation."""
    
    loan_id: str = Field(..., min_length=1, max_length=50)
    age: int = Field(..., ge=18, le=100, description="Borrower age in years")
    income: int = Field(..., ge=0, description="Annual income in USD")
    loan_amount: int = Field(..., ge=1000, le=1000000, description="Loan amount in USD")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    months_employed: int = Field(..., ge=0, le=600, description="Months employed")
    num_credit_lines: int = Field(..., ge=1, le=20, description="Number of credit lines")
    interest_rate: float = Field(..., ge=0.0, le=50.0, description="Interest rate percentage")
    loan_term: int = Field(..., ge=12, le=360, description="Loan term in months")
    dti_ratio: float = Field(..., ge=0.0, le=1.0, description="Debt-to-income ratio")
    education: str = Field(..., regex="^(High School|Bachelor's|Master's|PhD)$")
    employment_type: str = Field(..., regex="^(Full-time|Part-time|Self-employed|Unemployed)$")
    marital_status: str = Field(..., regex="^(Single|Married|Divorced)$")
    has_mortgage: str = Field(..., regex="^(Yes|No)$")
    has_dependents: str = Field(..., regex="^(Yes|No)$")
    loan_purpose: str = Field(..., regex="^(Home|Auto|Education|Business|Other)$")
    has_cosigner: str = Field(..., regex="^(Yes|No)$")
    
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
    
    class Config:
        schema_extra = {
            "example": {
                "loan_id": "LN123456",
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
        }

class DataIngestion:
    """Handles data ingestion from various sources with validation."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.supported_formats = ['csv', 'json', 'parquet']
        
    def load_data(self, 
                  source: Union[str, pd.DataFrame], 
                  source_type: str = 'csv',
                  validate: bool = True) -> pd.DataFrame:
        """
        Load data from various sources with optional validation.
        
        Args:
            source: Data source (file path or DataFrame)
            source_type: Type of source ('csv', 'json', 'parquet', 'dataframe')
            validate: Whether to validate data against schema
            
        Returns:
            Validated DataFrame
        """
        try:
            if isinstance(source, pd.DataFrame):
                df = source.copy()
            elif source_type == 'csv':
                df = pd.read_csv(source)
            elif source_type == 'json':
                df = pd.read_json(source)
            elif source_type == 'parquet':
                df = pd.read_parquet(source)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
            
            logger.info(f"Loaded {len(df)} records from {source_type} source")
            
            if validate:
                df = self._validate_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {source}: {str(e)}")
            raise
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate DataFrame against LoanApplication schema."""
        valid_rows = []
        invalid_indices = []
        
        for idx, row in df.iterrows():
            try:
                # Convert row to dict and validate
                row_dict = row.to_dict()
                loan_app = LoanApplication(**row_dict)
                valid_rows.append(row_dict)
            except Exception as e:
                invalid_indices.append((idx, str(e)))
                logger.warning(f"Invalid row {idx}: {str(e)}")
        
        if invalid_indices:
            logger.warning(f"Found {len(invalid_indices)} invalid records")
            for idx, error in invalid_indices[:10]:  # Log first 10 errors
                logger.warning(f"Row {idx}: {error}")
        
        valid_df = pd.DataFrame(valid_rows)
        logger.info(f"Validated {len(valid_df)} records out of {len(df)} total")
        
        return valid_df
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive data quality report."""
        report = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'duplicate_records': df.duplicated().sum(),
            'numeric_summary': df.describe().to_dict(),
            'categorical_summary': {}
        }
        
        # Categorical feature analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            report['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'value_counts': df[col].value_counts().head(10).to_dict()
            }
        
        return report

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'data_source': 'data/raw/train.csv',
        'validation_rules': {
            'min_age': 18,
            'max_loan_amount': 1000000
        }
    }
    
    # Initialize data ingestion
    ingestion = DataIngestion(config)
    
    # Load and validate data
    try:
        df = ingestion.load_data('data/raw/train.csv', source_type='csv', validate=True)
        quality_report = ingestion.get_data_quality_report(df)
        
        print(f"Successfully loaded {len(df)} validated records")
        print(f"Data quality: {quality_report['duplicate_records']} duplicates found")
        
    except Exception as e:
        print(f"Data ingestion failed: {str(e)}")
