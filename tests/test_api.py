"""
Tests for API endpoints
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.api import app
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints"""
    
    def test_health_check(self, client):
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_root_endpoint(self, client):
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestConfigEndpoints:
    """Tests for configuration endpoints"""
    
    def test_get_config(self, client):
        response = client.get("/api/v1/config")
        
        assert response.status_code == 200
        data = response.json()
        assert "ocr_engine" in data
        assert "languages" in data
    
    def test_get_ocr_engines(self, client):
        response = client.get("/api/v1/ocr-engines")
        
        assert response.status_code == 200
        data = response.json()
        assert "engines" in data
        assert "paddleocr" in data["engines"]
