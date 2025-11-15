"""Configuration module for Mango AI application."""
from .settings import (
    BASE_DIR,
    MODEL_PATHS,
    ASSETS_DIR,
    IMAGES_DIR,
    TEMP_DIR,
    TRAINING_DIR,
    APP_TITLE,
    APP_ICON,
    APP_LAYOUT,
    DISEASE_CLASS_NAMES,
    VARIETY_CLASS_NAMES,
    DAMAGE_CLASS_NAMES,
    IMAGE_SIZE,
    CONFIDENCE_THRESHOLD,
)

__all__ = [
    "BASE_DIR",
    "MODEL_PATHS",
    "ASSETS_DIR",
    "IMAGES_DIR",
    "TEMP_DIR",
    "TRAINING_DIR",
    "APP_TITLE",
    "APP_ICON",
    "APP_LAYOUT",
    "DISEASE_CLASS_NAMES",
    "VARIETY_CLASS_NAMES",
    "DAMAGE_CLASS_NAMES",
    "IMAGE_SIZE",
    "CONFIDENCE_THRESHOLD",
]

