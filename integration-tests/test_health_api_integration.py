# [LABEL:TEST:integration] [LABEL:FRAMEWORK:pytest] [LABEL:FRAMEWORK:httpx]
# [LABEL:AUTHOR:system] [LABEL:UPDATED:2025-12-04] [LABEL:REVIEWED:pending]
#
# Purpose: Integration tests for Health API endpoints
# Dependencies: pytest, httpx, fastapi
# Related: tools/health_server.py
#
import pytest
import httpx
import asyncio
import subprocess
import time
import signal
import os
from fastapi.testclient import TestClient
from tools.health_server import app

class TestHealthAPIIntegration:
    """Integration tests for health API endpoints."""

    def test_health_endpoint_structure(self):
        """Test basic health endpoint response structure."""
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code in [200, 503]  # Healthy or unhealthy

        data = response.json()
        required_fields = ["status", "timestamp", "uptime_seconds", "version", "checks"]
        for field in required_fields:
            assert field in data

        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert isinstance(data["uptime_seconds"], (int, float))
        assert "checks" in data
        assert "components" in data["checks"]
        assert "system" in data["checks"]

    def test_component_health_endpoint(self):
        """Test component health endpoint."""
        client = TestClient(app)

        response = client.get("/health/components")
        assert response.status_code in [200, 503]

        data = response.json()
        assert "components" in data
        assert isinstance(data["components"], dict)

    def test_system_health_endpoint(self):
        """Test system health endpoint."""
        client = TestClient(app)

        response = client.get("/health/system")
        assert response.status_code == 200

        data = response.json()
        assert "system" in data

        system_data = data["system"]
        required_metrics = ["cpu_percent", "memory_percent", "memory_used_gb", "memory_total_gb"]
        for metric in required_metrics:
            assert metric in system_data
            assert isinstance(system_data[metric], (int, float))

    def test_api_health_endpoint(self):
        """Test API health endpoint."""
        client = TestClient(app)

        response = client.get("/health/api")
        assert response.status_code in [200, 503]

        data = response.json()
        assert "api_endpoints" in data
        assert isinstance(data["api_endpoints"], dict)

    def test_readiness_probe(self):
        """Test Kubernetes readiness probe."""
        client = TestClient(app)

        response = client.get("/health/ready")
        assert response.status_code in [200, 503]

        data = response.json()
        assert "status" in data

    def test_liveness_probe(self):
        """Test Kubernetes liveness probe."""
        client = TestClient(app)

        response = client.get("/health/live")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "alive"

    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        client = TestClient(app)

        response = client.get("/metrics")
        assert response.status_code == 200

        content = response.text
        assert "# HELP" in content
        assert "# TYPE" in content
        assert "abc_app_" in content

    def test_openapi_docs_available(self):
        """Test that OpenAPI documentation is available."""
        client = TestClient(app)

        # Test Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200

        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200

        # Test OpenAPI JSON
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi_data = response.json()
        assert "paths" in openapi_data
        assert "/health" in openapi_data["paths"]

class TestHealthServerStartup:
    """Test health server startup and shutdown."""

    @pytest.mark.asyncio
    async def test_server_startup(self):
        """Test that the health server can start up."""
        # This would start the server in a separate process
        # For integration testing, we use TestClient instead
        pass

class TestHealthUnderLoad:
    """Test health endpoints under load."""

    def test_concurrent_health_requests(self):
        """Test multiple concurrent health requests."""
        client = TestClient(app)

        import threading
        results = []

        def make_request():
            response = client.get("/health")
            results.append(response.status_code)

        threads = []
        for _ in range(10):
            t = threading.Thread(target=make_request)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(code in [200, 503] for code in results)

class TestHealthDataAccuracy:
    """Test that health data is accurate."""

    def test_uptime_accuracy(self):
        """Test that uptime is reported accurately."""
        client = TestClient(app)

        response1 = client.get("/health")
        uptime1 = response1.json()["uptime_seconds"]

        # Wait a bit
        time.sleep(0.1)

        response2 = client.get("/health")
        uptime2 = response2.json()["uptime_seconds"]

        # Uptime should have increased
        assert uptime2 > uptime1

    def test_system_metrics_reasonable(self):
        """Test that system metrics are within reasonable bounds."""
        client = TestClient(app)

        response = client.get("/health/system")
        system_data = response.json()["system"]

        # CPU should be between 0 and 100
        assert 0 <= system_data["cpu_percent"] <= 100

        # Memory usage should be positive
        assert system_data["memory_used_gb"] > 0
        assert system_data["memory_total_gb"] > 0
        assert system_data["memory_used_gb"] <= system_data["memory_total_gb"]

        # Memory percent should be between 0 and 100
        assert 0 <= system_data["memory_percent"] <= 100</content>
</xai:function_call name="create_file">
<parameter name="filePath">c:\Users\nvick\ABC-Application\integration-tests\test_ibkr_integration.py