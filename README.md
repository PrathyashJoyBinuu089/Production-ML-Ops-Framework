# Production MLOps Framework

An enterprise-grade, modular framework for building, training, and serving machine learning models with production-ready standards.

## 🏗️ Architecture Overview

The framework is structured to separate concerns between experimentation, automated training, and high-performance serving.

```text
Production-ML-Ops-Framework/
├── src/
│   ├── api/               # FastAPI Serving Layer
│   │   └── main.py        # API implementation
│   └── pipelines/         # Training & Evaluation Pipelines
│       └── training_pipeline.py
├── infrastructure/        # Deployment Configuration
│   └── Dockerfile         # Multi-stage build for API
├── models/                # Model Artifact Registry (Git ignored)
├── tests/                 # Automated Testing
│   └── test_api.py
├── requirements.txt       # Project Dependencies
└── README.md              # Project Documentation
```

## 🔄 MLOps Lifecycle

### 1. Experimentation & Development
Data Scientists develop models using the modular components in `src/pipelines`. Logging is handled by `loguru` to ensure traceablity.

### 2. Automated Training
The `training_pipeline.py` script handles data ingestion, preprocessing, training, and evaluation. It follows a consistent logic for artifact versioning:
- **Timestamped artifacts**: Saved as `{model_name}_{timestamp}.joblib` for auditability.
- **Latest pointer**: Updated as `{model_name}_latest.joblib` for easy deployment consumption.

### 3. Model Serving (API)
The FastAPI application in `src/api` provides:
- **Health Checks**: `/health` endpoint for monitoring and orchestration (e.g., Kubernetes Liveness/Readiness probes).
- **Validation**: Strict Pydantic schemas for input/output verification.
- **Performance**: High-concurrency support using Uvicorn.

### 4. Continuous Integration & Deployment (CI/CD)
The project includes a multi-stage `Dockerfile` that minimizes image size and protects source code by only including necessary artifacts in the final production stage.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### Installation
```bash
# Clone the repository
git clone <repo-url>
cd Production-ML-Ops-Framework

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training a Model
```bash
export PYTHONPATH=$PYTHONPATH:.
python src/pipelines/training_pipeline.py
```

### Starting the Serving API
```bash
# Ensure a model exists in models/iris_classifier_latest.joblib
uvicorn src.api.main:app --reload
```

### Running Tests
```bash
pytest
```

## 🛠️ Tech Stack
- **FastAPI**: Modern, fast (high-performance) web framework for building APIs.
- **Scikit-Learn**: Simple and efficient tools for predictive data analysis.
- **Loguru**: Python logging made (stupidly) simple.
- **Joblib**: Lightweight pipelining with Python objects.
- **Pydantic**: Data validation and settings management using Python type hints.
- **Pytest**: A mature full-featured Python testing tool.
