# 🍋 AI-Driven Precision Farming for Mangoes

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

A unified AI-powered system that acts as the "brain" for the entire mango lifecycle, integrating classification, detection, prediction, and grading modules into a central decision support system.

---

## 📜 Introduction and Problem Statement

India is the world's largest mango producer, yet it struggles with **low productivity and quality**. The core problem is the reliance on **manual inspection** in both farming and production industries, which is:
- ⏱️ **Slow** - Time-consuming manual processes
- 🔄 **Inconsistent** - Varies between inspectors
- ❌ **Error-prone** - Human judgment limitations

This project proposes a **unified AI system** that integrates multiple models to handle:
- 🍋 **Classification** - Identifying mango varieties
- 🌿 **Detection** - Detecting diseases with interpretability
- 📊 **Prediction** - Predicting ripeness and sensory attributes
- 🎯 **Grading** - Classifying fruit quality based on UN/ECE standards

All modules feed into a central decision support system accessible through a user-friendly Streamlit interface.

---

## 🎯 Key Objectives

The project has five main objectives:

1. **Develop a model to classify different Indian mango varieties** - Identify 15 popular Indian mango varieties from images
2. **Create an interpretable model to detect mango diseases** - Detect 8 different leaf diseases with visual explanations
3. **Predict mango ripeness and integrate it with sensory analysis** - Non-destructive ripeness assessment with sensory attribute predictions
4. **Identify fruit grade based on UN/ECE standards for damage** - Grade mangoes into Extra Class, Class I, and Class II
5. **Integrate all modules into a single, user-friendly software system** - Unified Streamlit-based platform

---

## 🔬 Methodology and Results

### 1. 🍋 Fruit Variety Classification

**Methodology:**
- Dataset: 15 Indian mango varieties (Alphonso, Ambika, Amrapali, Banganpalli, Chausa, Dasheri, Himsagar, Kesar, Langra, Malgova, Mallika, Neelam, Raspuri, Totapuri, Vanraj)
- Process:
  1. Image preprocessing and resizing
  2. HSV color filtering to isolate mango regions
  3. Texture and edge detection
  4. Classification using **RepVGG-A1** model
- Model: RepVGG-A1 (also tested ConvNeXt and TinyViT)

**Results:**
- High accuracy across all tested models
- ANOVA test confirmed no statistically significant difference between models
- Robust performance on diverse mango images

---

### 2. 🌿 Disease Detection

**Methodology:**
- Model: **ConvNeXt-Tiny** backbone
- Classes: 8 diseases + healthy category
  - Anthracnose
  - Bacterial Canker
  - Cutting Weevil
  - Die Back
  - Gall Midge
  - Powdery Mildew
  - Sooty Mould
  - Healthy
- Interpretability: Class Activation Map (CAM) techniques
  - Compared: Grad-CAM, Eigen-CAM, **Layer-CAM**
  - Generates heatmaps showing diseased regions

**Results:**
- **Layer-CAM** was the top performer:
  - Highest Average IoU: **0.93**
  - Fastest runtime: **0.2418s**
  - Most precise localization of disease spots

---

### 3. 📊 Ripeness Prediction

**Methodology:**
- Non-destructive method analyzing 8 parameters:
  - RGB color values (R, G, B)
  - Weight (W)
  - Volume (V)
  - Weight/Volume ratio (W/C)
  - Storage Time
  - Days After Flowering (DAFS)
  - Titratable Acidity (TA)
  - Total Soluble Solids (TSS)
- Model: **CatBoost** regressor
- Interpretability: **SHAP** values for feature importance

**Results:**
- TSS prediction: **R² = 0.75**
- TSS/TA ratio: **R² = 0.61** (where AdaBoost failed)
- Most critical predictors: **Color and temporal features** (Days After Flowering)
- Integrated sensory attribute predictions (Taste, Flavour, Appearance, Overall Acceptability)

---

### 4. 🎯 Fruit Grading

**Methodology:**
- Hybrid model for UN/ECE standard grading:
  - **Extra Class** - Highest quality
  - **Class I** - Good quality
  - **Class II** - Acceptable quality
- Feature extraction:
  - Deep features from **CoaT** (transformer) model
  - Handcrafted features: color histogram, texture (Haralick), shape (Hu moments)
- Classification: **Random Forest**
- Interpretability: **LIME** for explaining grading decisions

**Results:**
- Validation accuracy: **89%**
- Cohen's Kappa score: **0.794** (substantial agreement)
- Robust grading based on damage assessment

---

## 💻 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/Hemanth-310/AI-Driven-Precision-Farming-for-Mangoes
cd AI-Driven-Precision-Farming-for-Mangoes
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Model Paths

Update the model paths in `src/config/settings.py` to point to your actual model files:

```python
MODEL_PATHS = {
    "disease_detection": "path/to/convnext_disease.pth",
    "fruit_variety": "path/to/repvgg_mango_classifier.pth",
    "fruit_grading": "path/to/random_forest_model.pkl",
    "ripeness": "path/to/catboost_{target}.cbm",
}
```

### Step 4: Organize Model Files

Place your model files in the `models/` directory structure:

```
models/
├── disease_detection/
│   └── convnext_disease.pth
├── fruit_variety/
│   └── repvgg_mango_classifier.pth
├── fruit_grading/
│   └── random_forest_model.pkl
└── ripeness/
    └── catboost_TSS.cbm
```

---

## 🚀 Usage

### Running the Application

```bash
streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`

### Features

1. **🏠 Welcome Page** - Introduction and mango cultivation information
2. **📖 About** - Overview of the Mangoo AI system
3. **🍋 Variety Classification** - Upload a mango image to identify its variety
4. **🌿 Disease Detection** - Upload a leaf image to detect diseases with heatmap visualization
5. **🍈 Ripeness Analysis** - Upload a mango image with parameters (storage time, DAFS, weight) to predict ripeness and sensory attributes
6. **🍂 Damage Grading** - Upload a mango image to grade it according to UN/ECE standards

---

## 📁 Project Structure

```
mango_code/
├── src/                          # Main application source code
│   ├── app.py                    # Main Streamlit application
│   ├── config/                   # Configuration module
│   │   ├── __init__.py
│   │   └── settings.py           # All configuration settings and paths
│   ├── pages/                    # Streamlit page modules
│   │   ├── welcome.py            # Welcome/home page
│   │   ├── about.py              # About page
│   │   ├── variety.py            # Mango variety classification
│   │   ├── disease.py            # Disease detection
│   │   ├── ripeness.py           # Ripeness analysis
│   │   └── damage.py             # Fruit damage/grading
│   └── utils/                    # Utility modules
│       ├── ripeness.py           # Ripeness detection utilities
│       ├── disease_detection.py  # Disease detection utilities
│       ├── fruit_grading.py      # Fruit grading utilities
│       └── variety_prediction.py # Variety prediction utilities
├── assets/                       # Static assets
│   └── images/                  # Image assets for the UI
├── models/                       # Model files (not in git)
│   ├── disease_detection/
│   ├── fruit_variety/
│   ├── fruit_grading/
│   └── ripeness/
├── training/                     # Training scripts (for reference)
│   ├── disease_detector/
│   ├── fruit_grading/
│   ├── mango_classifier/
│   └── mango_ripeness/
├── temp/                         # Temporary files directory
├── main.py                       # Main entry point
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## 📦 Dependencies

Key dependencies include:

- **streamlit** - Web application framework
- **torch** & **torchvision** - Deep learning framework
- **timm** - Pre-trained model library
- **catboost** - Gradient boosting for ripeness prediction
- **opencv-python** - Image processing
- **numpy** & **pandas** - Data manipulation
- **matplotlib** - Visualization
- **grad-cam** - Model interpretability (Layer-CAM)
- **mahotas** - Texture feature extraction
- **scikit-learn** - Machine learning utilities
- **joblib** - Model serialization

See `requirements.txt` for the complete list.

---

## 🔧 Configuration

All configuration is centralized in `src/config/settings.py`:

- **Model paths** - Update to point to your model files
- **Class names** - Disease, variety, and damage class definitions
- **Image settings** - Image size, confidence thresholds
- **Directory paths** - Assets, models, temp directories

---

## 💬 Discussion and Conclusion

### Strengths

- ✅ **Unified System** - Acts as a "control center" integrating all modules
- ✅ **Advanced Preprocessing** - Reduces errors through robust image processing
- ✅ **User-Friendly Platform** - Streamlit-based interface accessible to farmers
- ✅ **Interpretability** - CAM techniques and LIME provide explainable AI
- ✅ **High Performance** - Strong accuracy across all modules

### Limitations

- ⚠️ **Image-Only Input** - Currently relies only on image data
- ⚠️ **Model Dependencies** - Requires pre-trained model files

### Future Work

- 🔮 **IoT Integration** - Integrate sensors for soil, weather, and environmental data
- ☁️ **Cloud Deployment** - Deploy on cloud platforms for scalability
- 📱 **Edge Deployment** - Mobile/edge device support for offline usability
- 🔄 **Real-time Monitoring** - Continuous monitoring and alerting system
- 📈 **Analytics Dashboard** - Advanced analytics and reporting features

### Impact

This system aims to:
- 🌾 **Reduce waste** - Better quality assessment and grading
- 📈 **Increase productivity** - Automated, consistent inspection
- 🌍 **Improve competitiveness** - Enhanced quality of Indian mango exports
- 👨‍🌾 **Empower farmers** - Accessible AI tools for actionable recommendations

---

## 📝 Citation

If you use this project in your research, please cite:

```
Mango AI - Intelligent Decision Support System for Mango Cultivation
A unified AI system integrating variety classification, disease detection, 
ripeness prediction, and fruit grading for mango cultivation.
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

[Add author information]

---

## 🙏 Acknowledgments

- Indian mango farmers and agricultural researchers
- Open-source ML community
- Dataset contributors

---

**Made with ❤️ for mango farmers and agricultural innovation**

