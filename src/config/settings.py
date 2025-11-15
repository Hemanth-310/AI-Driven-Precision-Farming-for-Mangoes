"""
Configuration settings for the Mango AI application.
Update model paths and other settings here.
"""
import os
from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent.parent

# Model paths - Update these to point to your actual model files
MODEL_PATHS = {
    "disease_detection": os.path.join(
        BASE_DIR, "models", "disease_detection", "convnext_disease.pth"
    ),
    "fruit_variety": os.path.join(
        BASE_DIR, "models", "fruit_variety", "repvgg_mango_classifier.pth"
    ),
    "fruit_grading": os.path.join(
        BASE_DIR, "models", "fruit_grading", "random_forest_model.pkl"
    ),
    "ripeness": os.path.join(
        BASE_DIR, "models", "ripeness", "catboost_{target}.cbm"
    ),
}

# Asset paths
ASSETS_DIR = BASE_DIR / "assets"
IMAGES_DIR = ASSETS_DIR / "images"

# Temporary files directory
TEMP_DIR = BASE_DIR / "temp"

# Training scripts directory
TRAINING_DIR = BASE_DIR / "training"

# Application settings
APP_TITLE = "Mangoo AI"
APP_ICON = "🍋"
APP_LAYOUT = "wide"

# Model settings
DISEASE_CLASS_NAMES = [
    'Anthracnose', 'Bacterial Canker', 'Cutting Wevil', 'Die Back',
    'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould'
]

VARIETY_CLASS_NAMES = [
    'alphonsa', 'Ambika', 'Amrapali', 'Banganpalli', 'Chausa', 'Dasheri',
    'Himsagar', 'Kesar', 'Langra', 'Malgova', 'Mallika', 'Neelam', 
    'Raspuri', 'totapuri', 'Vanraj'
]

DAMAGE_CLASS_NAMES = {
    0: 'Undamaged',
    1: 'Mild Damage',
    2: 'Severe Damage'
}

# Image processing settings
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.50

# Create directories if they don't exist
for path in [TEMP_DIR, IMAGES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

