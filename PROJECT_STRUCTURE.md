# Project Structure

This document describes the restructured Mango AI project organization.

## Directory Structure

```
mango_code/
├── src/                          # Main application source code
│   ├── __init__.py
│   ├── app.py                    # Main Streamlit application entry point
│   ├── config/                   # Configuration module
│   │   ├── __init__.py
│   │   └── settings.py           # All configuration settings and paths
│   ├── pages/                    # Streamlit page modules
│   │   ├── __init__.py
│   │   ├── welcome.py            # Welcome/home page
│   │   ├── about.py              # About page
│   │   ├── variety.py            # Mango variety classification page
│   │   ├── disease.py            # Disease detection page
│   │   ├── ripeness.py           # Ripeness analysis page
│   │   └── damage.py             # Fruit damage/grading page
│   └── utils/                    # Utility modules
│       ├── __init__.py
│       ├── ripeness.py           # Ripeness detection utilities
│       ├── disease_detection.py  # Disease detection utilities
│       ├── fruit_grading.py      # Fruit grading utilities
│       └── variety_prediction.py # Variety prediction utilities
├── assets/                       # Static assets
│   └── images/                   # Image assets for the UI
├── models/                       # Model files (not in git)
│   ├── disease_detection/
│   ├── fruit_variety/
│   ├── fruit_grading/
│   └── ripeness/
├── training/                     # Training scripts (kept separate)
│   ├── disease_detector/
│   ├── fruit_grading/
│   ├── mango_classifier/
│   └── mango_ripeness/
├── temp/                         # Temporary files directory
├── main.py                       # Main entry point (run: streamlit run main.py)
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── PROJECT_STRUCTURE.md         # This file
```

## Key Improvements

1. **Modular Structure**: Code is organized into logical modules (pages, utils, config)
2. **Configuration Management**: All paths and settings centralized in `src/config/settings.py`
3. **Separation of Concerns**: UI pages separated from business logic utilities
4. **Clean Imports**: Proper package structure with `__init__.py` files
5. **Asset Organization**: Images moved to dedicated `assets/` directory
6. **Training Scripts**: Kept separate in `training/` directory
7. **Temporary Files**: Centralized in `temp/` directory

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

## Configuration

Update model paths in `src/config/settings.py` to point to your actual model files.

## Notes

- Model files should be placed in the `models/` directory structure
- Temporary files are automatically created in `temp/` directory
- All image assets should be in `assets/images/`
- Training scripts remain in `training/` for reference

