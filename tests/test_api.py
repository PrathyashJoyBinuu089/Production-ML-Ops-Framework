"""
Unit tests for the FastAPI model serving service.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    """Tests the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    # Note: If no model is trained yet, this might return unhealthy, 
    # but the endpoint itself should respond.
    assert "status" in response.json()

def test_prediction_invalid_input():
    """Tests prediction with invalid input data."""
    response = client.post(
        "/predict",
        json={"features": "invalid"}
    )
    assert response.status_code == 422  # Unprocessable Entity (Pydantic error)

@pytest.mark.skip(reason="Requires a trained model artifact to be present.")
def test_prediction_success():
    """Tests prediction with valid features (requires model)."""
    payload = {
        "features": [[5.1, 3.5, 1.4, 0.2]]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert isinstance(response.json()["predictions"], list)
