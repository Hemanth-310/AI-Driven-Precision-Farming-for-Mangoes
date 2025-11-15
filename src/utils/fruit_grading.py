"""
Fruit grading utilities for mango damage detection.
Uses Random Forest model with CoaT feature extraction.
"""
import cv2
import numpy as np
import torch
import timm
from PIL import Image
import mahotas
import joblib
from torchvision import transforms

from ..config import MODEL_PATHS, DAMAGE_CLASS_NAMES, IMAGE_SIZE

# Setup CoaT model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
coat = timm.create_model("coat_lite_small", pretrained=True)
coat.head = torch.nn.Identity()
coat.eval()
coat.to(DEVICE)

# Transform for CoaT
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Random Forest model
_rf_model = None


def _load_rf_model():
    """Lazy load the Random Forest model."""
    global _rf_model
    if _rf_model is None:
        model_path = MODEL_PATHS["fruit_grading"]
        _rf_model = joblib.load(model_path)
    return _rf_model


def color_histogram(img):
    """
    Extract color histogram features from image.
    
    Args:
        img: RGB image as numpy array
    
    Returns:
        Normalized histogram features
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def texture_features(img_gray):
    """
    Extract texture features using Haralick features.
    
    Args:
        img_gray: Grayscale image as numpy array
    
    Returns:
        Mean Haralick features
    """
    return mahotas.features.haralick(img_gray).mean(axis=0)


def shape_features(img_gray):
    """
    Extract shape features using Hu moments.
    
    Args:
        img_gray: Grayscale image as numpy array
    
    Returns:
        Flattened Hu moments
    """
    moments = cv2.moments(img_gray)
    return cv2.HuMoments(moments).flatten()


def extract_cnn_features(image_tensor):
    """
    Extract CNN features using CoaT model.
    
    Args:
        image_tensor: Preprocessed image tensor
    
    Returns:
        Flattened feature vector
    """
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        features = coat(image_tensor)
        return features.cpu().numpy().flatten()


def extract_all_features(img_path):
    """
    Extract all features (handcrafted + CNN) from image.
    
    Args:
        img_path: Path to image file
    
    Returns:
        Concatenated feature vector
    """
    img_pil = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.array(img_pil)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    handcrafted = np.concatenate([
        color_histogram(img_np),
        texture_features(img_gray),
        shape_features(img_gray)
    ])

    img_tensor = transform(img_pil)
    cnn_feat = extract_cnn_features(img_tensor)

    return np.concatenate([cnn_feat, handcrafted])


def predict_damage(image_path):
    """
    Predict damage level from mango image.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Tuple of (img_rgb, pred_label, pred_class)
        - img_rgb: RGB image as numpy array
        - pred_label: Predicted damage class name
        - pred_class: Predicted class index
    """
    rf_model = _load_rf_model()
    features = extract_all_features(image_path)
    pred_rf = rf_model.predict([features])[0]
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be read.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb, DAMAGE_CLASS_NAMES[pred_rf], pred_rf

