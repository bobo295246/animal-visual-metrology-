"""
Automated Integration and Unit Tests for the FastAPI Metrology Service.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

client = TestClient(app)


def test_dashboard_endpoint():
    """Verify HTML Dashboard returns 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert "COCO Animal Segmentation" in response.text


def test_auth_login_success_admin():
    """Verify Admin authentication returns valid JWT token."""
    response = client.post(
        "/api/v1/auth/login",
        data={"username": settings.TEST_ADMIN_USERNAME, "password": settings.TEST_ADMIN_PASSWORD},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["role"] == "admin"


def test_auth_login_success_tester():
    """Verify Tester authentication returns valid JWT token."""
    response = client.post(
        "/api/v1/auth/login",
        data={"username": settings.TEST_USER_USERNAME, "password": settings.TEST_USER_PASSWORD},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["role"] == "tester"


def test_auth_login_failure():
    """Verify invalid credentials return 401 Unauthorized."""
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "wrong_user", "password": "wrong_password"},
    )
    assert response.status_code == 401


def test_get_sample_measurements_csv():
    """Verify downloading pre-generated CSV returns CSV content."""
    response = client.get("/api/v1/metrology/sample-measurements-csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "intra_eye_distance_px" in response.text


def test_get_benchmark_report():
    """Verify benchmark verification endpoint returns MAE, RMSE, MAPE."""
    response = client.get("/api/v1/metrology/benchmark-verification")
    assert response.status_code == 200
    data = response.json()
    assert "mean_absolute_error_px" in data
    assert "root_mean_squared_error_px" in data
    assert "mean_absolute_percentage_error_pct" in data
    assert data["total_animals_evaluated"] >= 2
    assert data["accuracy_rate_pct"] > 85.0


def test_analyze_image_authenticated():
    """Verify uploading an image returns segmentation masks and distances."""
    # Obtain JWT
    login_resp = client.post(
        "/api/v1/auth/login",
        data={"username": settings.TEST_ADMIN_USERNAME, "password": settings.TEST_ADMIN_PASSWORD},
    )
    token = login_resp.json()["access_token"]

    sample_img_path = Path("data/coco_samples/coco_000000555705_two_cats_shoes.jpg")
    with open(sample_img_path, "rb") as f:
        response = client.post(
            "/api/v1/metrology/analyze-image",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": (sample_img_path.name, f, "image/jpeg")},
            data={"scale_mm_per_px": 0.25},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total_animals_detected"] >= 2
    assert len(data["animals"]) >= 2

    # Check each animal has eyes and intra distance
    for animal in data["animals"]:
        assert "left_eye" in animal
        assert "right_eye" in animal
        assert animal["intra_eye_distance_px"] > 0

    # Check inter-animal right-eye distances exist
    assert len(data["inter_animal_right_eye_distances"]) >= 1
    assert data["inter_animal_right_eye_distances"][0]["distance_px"] > 0
