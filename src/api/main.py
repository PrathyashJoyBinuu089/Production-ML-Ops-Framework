"""
FastAPI application for model serving.
Author: Senior ML Engineer
"""
import os
import joblib
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

# Model Registry
class ModelRegistry:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """Loads the model from disk."""
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found at {self.model_path}")
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def predict(self, features: List[List[float]]):
        """Generates predictions from features."""
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        
        try:
            return self.model.predict(features).tolist()
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

# Pydantic Schemas
class PredictionInput(BaseModel):
    """Features input schema."""
    features: List[List[float]] = Field(..., example=[[5.1, 3.5, 1.4, 0.2]])

class PredictionOutput(BaseModel):
    """Prediction output schema."""
    predictions: List[int]
    model_version: Optional[str] = "1.0.0"

# Lifespan context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Starting up FastAPI application...")
    
    # Locate model path relative to app
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "models", "iris_classifier_latest.joblib")
    
    app.state.model_registry = ModelRegistry(model_path)
    
    # Try to load model, if it fails, log it but don't prevent startup if you want health check to work
    try:
        app.state.model_registry.load_model()
    except Exception as e:
        logger.warning(f"Warning: Model could not be loaded on startup. Health check will report as unhealthy. Error: {str(e)}")
    
    yield
    # Shutdown logic
    logger.info("Shutting down FastAPI application...")

app = FastAPI(
    title="Production ML Serving API",
    description="Enterprise-grade FastAPI service for ML model predictions.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", status_code=200)
async def health_check():
    """Health check endpoint to ensure the service and model are ready."""
    if hasattr(app.state, "model_registry") and app.state.model_registry.model is not None:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Prediction endpoint for Iris classification."""
    if not hasattr(app.state, "model_registry") or app.state.model_registry.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or unavailable.")
    
    try:
        predictions = app.state.model_registry.predict(input_data.features)
        return PredictionOutput(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
