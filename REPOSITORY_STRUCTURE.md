# Repository Structure

```
loan-default-prediction/
├── README.md                           # Project overview and quickstart
├── PRD.md                             # Product requirements document
├── architecture.md                     # System design and architecture
├── docker-compose.yml                  # Local development environment
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Continuous integration
│       ├── model-training.yml          # Automated model retraining
│       └── deploy.yml                 # Deployment pipeline
├── data/
│   ├── raw/                           # Raw dataset files
│   ├── processed/                     # Cleaned and preprocessed data
│   └── external/                      # External data sources
├── models/
│   ├── production/                    # Production-ready models
│   ├── staging/                       # Staging models
│   └── experiments/                   # Experimental models
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py               # Data loading and validation
│   │   ├── preprocessing.py           # Feature engineering pipeline
│   │   └── validation.py              # Data quality checks
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                   # Model training script
│   │   ├── evaluate.py                # Model evaluation
│   │   ├── explain.py                 # Model explainability
│   │   └── export.py                  # Model export to ONNX
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI application
│   │   ├── schemas.py                 # Pydantic models
│   │   ├── endpoints/
│   │   │   ├── __init__.py
│   │   │   ├── predict.py             # Prediction endpoint
│   │   │   ├── explain.py             # Explainability endpoint
│   │   │   └── health.py              # Health check endpoint
│   │   └── middleware/
│   │       ├── __init__.py
│   │       ├── auth.py                 # Authentication middleware
│   │       └── logging.py              # Request logging
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Prometheus metrics
│   │   └── drift_detection.py         # Data drift monitoring
│   └── utils/
│       ├── __init__.py
│       ├── config.py                  # Configuration management
│       └── helpers.py                 # Utility functions
├── frontend/
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx             # Root layout
│   │   │   ├── page.tsx               # Dashboard page
│   │   │   ├── predict/
│   │   │   │   └── page.tsx           # Prediction interface
│   │   │   └── explain/
│   │   │       └── page.tsx           # Explainability view
│   │   ├── components/
│   │   │   ├── ui/                    # Reusable UI components
│   │   │   ├── forms/                 # Form components
│   │   │   └── charts/                # Data visualization
│   │   ├── lib/
│   │   │   ├── api.ts                 # API client
│   │   │   └── utils.ts               # Utility functions
│   │   └── types/
│   │       └── index.ts               # TypeScript type definitions
├── tests/
│   ├── unit/
│   │   ├── test_data_ingestion.py
│   │   ├── test_preprocessing.py
│   │   ├── test_models.py
│   │   └── test_api.py
│   ├── integration/
│   │   ├── test_api_endpoints.py
│   │   └── test_model_pipeline.py
│   └── fixtures/
│       ├── sample_data.csv
│       └── mock_models/
├── scripts/
│   ├── setup.sh                       # Environment setup
│   ├── train_model.sh                 # Model training script
│   ├── deploy.sh                      # Deployment script
│   └── monitor.sh                     # Monitoring setup
├── docs/
│   ├── api/                           # API documentation
│   ├── deployment/                    # Deployment guides
│   └── development/                   # Development setup
├── infrastructure/
│   ├── docker/
│   │   ├── Dockerfile.api             # API service Dockerfile
│   │   ├── Dockerfile.frontend        # Frontend Dockerfile
│   │   └── Dockerfile.training        # Training Dockerfile
│   ├── kubernetes/
│   │   ├── namespace.yaml
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── ingress.yaml
│   └── terraform/
│       ├── main.tf                    # AWS/GCP infrastructure
│       ├── variables.tf
│       └── outputs.tf
├── config/
│   ├── model.yaml                     # Model configuration
│   ├── training.yaml                  # Training parameters
│   └── deployment.yaml                # Deployment settings
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project configuration
├── .env.example                       # Environment variables template
├── .gitignore
├── .pre-commit-config.yaml            # Pre-commit hooks
└── README.md
```

## Key Files Description

### Core ML Pipeline
- **src/data/ingestion.py**: Handles data loading from various sources with validation
- **src/models/train.py**: Orchestrates model training with hyperparameter optimization
- **src/models/export.py**: Converts trained models to ONNX format for production
- **src/api/main.py**: FastAPI application with prediction and explainability endpoints

### Frontend Application
- **frontend/src/app/page.tsx**: Main dashboard with risk assessment interface
- **frontend/src/components/forms/LoanForm.tsx**: Loan application form with validation
- **frontend/src/components/charts/RiskChart.tsx**: Visualization of risk factors

### Infrastructure and Deployment
- **docker-compose.yml**: Local development with all services
- **infrastructure/kubernetes/**: Production deployment manifests
- **.github/workflows/**: CI/CD pipelines for automated testing and deployment

### Testing and Quality
- **tests/**: Comprehensive test suite with unit and integration tests
- **pyproject.toml**: Code quality configuration (black, ruff, mypy)
- **.pre-commit-config.yaml**: Pre-commit hooks for code quality
