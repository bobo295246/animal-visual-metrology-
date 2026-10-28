"""Application configuration settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "COCO Animal Segmentation & Visual Metrology API"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    SECRET_KEY: str = "super-secret-jwt-key-change-in-production-2026"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120

    # Test Account Credentials
    TEST_ADMIN_USERNAME: str = "admin"
    TEST_ADMIN_PASSWORD: str = "AdminPassword123!"
    TEST_USER_USERNAME: str = "metrology_tester"
    TEST_USER_PASSWORD: str = "TesterPassword123!"

    # YOLO & Metrology settings
    YOLO_SEG_MODEL: str = "yolov8n-seg.pt"
    DEFAULT_EYE_METHOD: str = "vitpose"
    YOLO_CONF_THRESHOLD: float = 0.35
    YOLO_IOU_THRESHOLD: float = 0.45
    DEFAULT_PIXEL_TO_MM_SCALE: float = 0.25

    DATA_DIR: str = "data"
    OUTPUT_DIR: str = "outputs"
    CSV_EXPORT_PATH: str = "data/sample_measurements.csv"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
