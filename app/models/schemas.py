"""Pydantic schemas for requests and responses."""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    role: str
    username: str


class LoginRequest(BaseModel):
    username: str = Field(..., description="Username for login")
    password: str = Field(..., description="Password for login")


class EyeDetail(BaseModel):
    centroid: Tuple[float, float]
    bbox: List[int]
    area: float
    polygon: List[List[float]]


class AnimalDetail(BaseModel):
    animal_id: int
    species: str
    confidence: float
    bbox: List[int]
    body_polygon: List[List[float]]
    left_eye: EyeDetail
    right_eye: EyeDetail
    intra_eye_distance_px: float
    intra_eye_distance_mm: Optional[float] = None


class InterRightEyePair(BaseModel):
    pair: str
    animal_1_id: int
    animal_2_id: int
    animal_1_species: str
    animal_2_species: str
    animal_1_right_eye: Tuple[float, float]
    animal_2_right_eye: Tuple[float, float]
    distance_px: float
    distance_mm: Optional[float] = None


class MetrologyAnalysisResponse(BaseModel):
    image_name: str
    image_shape: List[int]
    eye_segmentation_method: Optional[str] = "vitpose"
    total_animals_detected: int
    annotated_image_url: Optional[str] = None
    animals: List[AnimalDetail]
    inter_animal_right_eye_distances: List[InterRightEyePair]


class BenchmarkVerificationResponse(BaseModel):
    test_image: str
    total_animals_evaluated: int
    mean_absolute_error_px: float
    root_mean_squared_error_px: float
    mean_absolute_percentage_error_pct: float
    accuracy_rate_pct: float
    evaluations: List[Dict[str, Any]]
