"""Typed settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DetectionModel = Literal["hog", "cnn"]


class Settings(BaseSettings):
    """Application configuration.

    Values come from the process environment, falling back to a local ``.env``.
    Unknown keys are ignored so that unrelated variables in the shell do not
    break startup.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Face Recognition API"
    version: str = "0.1.0"
    environment: Environment = "development"

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: LogLevel = "INFO"

    cors_origins: list[str] = Field(default_factory=lambda: ["http://127.0.0.1:8000"])

    # ---- Uploads ----
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, gt=0)
    # Guards against decompression bombs: a small file can decode to a huge array.
    max_image_pixels: int = Field(default=40_000_000, gt=0)
    min_image_edge: int = Field(default=32, gt=0)

    # ---- Recognition ----
    gallery_db_path: Path = Path("data/gallery.db")
    # Euclidean distance below which two embeddings are the same person.
    # 0.6 is dlib's published operating point for this model.
    match_tolerance: float = Field(default=0.6, gt=0.0, le=1.0)
    detection_model: DetectionModel = "hog"
    # Times to upsample before detecting. Higher finds smaller faces, costs time.
    detection_upsample: int = Field(default=1, ge=0, le=4)
    # Images are shrunk so their longest edge is at most this, before
    # detection. Detection cost scales with pixel count, so this bounds
    # per-frame latency regardless of the camera or photo resolution.
    detection_max_edge: int = Field(default=640, ge=0)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string, since env vars cannot hold lists."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, parsed once."""
    return Settings()
