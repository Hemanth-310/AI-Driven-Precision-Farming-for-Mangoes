"""
Ripeness detection utilities for mango analysis.
Handles RGB processing, TSS prediction, and sensory attribute estimation.
"""
import cv2
import numpy as np
from catboost import CatBoostRegressor
import os
import matplotlib.pyplot as plt
import streamlit as st

from ..config import MODEL_PATHS


class ImageProcessing:
    """Class for processing mango images and extracting RGB values."""
    
    def __init__(self, image_path):
        """
        Initialize ImageProcessing with an image path.
        
        Args:
            image_path: Path to the mango image file
        """
        self.image_path = image_path

    def extract_rgb_values(self):
        """
        Extract RGB values from top, center, and bottom regions of the image.
        
        Returns:
            Tuple of (top_region, center_region, bottom_region, 
                     top_rgb, center_rgb, bottom_rgb)
        """
        img = cv2.imread(self.image_path)
        if img is None:
            print("Error loading image.")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        top = img[0:h // 3, :]
        center = img[h // 3:2 * h // 3, :]
        bottom = img[2 * h // 3:, :]

        def get_avg(region):
            return np.mean(region, axis=(0, 1))

        return top, center, bottom, get_avg(top), get_avg(center), get_avg(bottom)

    def classify_rgb_stage(self):
        """
        Classify ripeness stage based on RGB values.
        
        Returns:
            String: "Ripe", "Unripe", "Mid-Ripe", or "Unknown"
        """
        top, center, bottom, top_rgb, center_rgb, bottom_rgb = self.extract_rgb_values()
        if top_rgb is None:
            return "Unknown"

        avg_r = (top_rgb[0] + center_rgb[0] + bottom_rgb[0]) / 3
        avg_g = (top_rgb[1] + center_rgb[1] + bottom_rgb[1]) / 3
        avg_b = (top_rgb[2] + center_rgb[2] + bottom_rgb[2]) / 3

        if avg_r > avg_g and avg_r > avg_b:
            return "Ripe"
        elif avg_g > avg_r and avg_g > avg_b:
            return "Unripe"
        else:
            return "Mid-Ripe"

    def get_avg_rgb(self):
        """
        Get average RGB values across all regions.
        
        Returns:
            Tuple of (avg_r, avg_g, avg_b) or None if image can't be loaded
        """
        top, center, bottom, top_rgb, center_rgb, bottom_rgb = self.extract_rgb_values()
        if top_rgb is None:
            return None
        avg_r = (top_rgb[0] + center_rgb[0] + bottom_rgb[0]) / 3
        avg_g = (top_rgb[1] + center_rgb[1] + bottom_rgb[1]) / 3
        avg_b = (top_rgb[2] + center_rgb[2] + bottom_rgb[2]) / 3
        return avg_r, avg_g, avg_b

    def plot_rgb_with_image(self):
        """
        Create a visualization of the image with RGB values overlaid.
        
        Returns:
            matplotlib Figure object or None if image can't be loaded
        """
        img = cv2.imread(self.image_path)
        if img is None:
            print("Error loading image.")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        top = img[0:h // 3, :]
        center = img[h // 3:2 * h // 3, :]
        bottom = img[2 * h // 3:, :]

        top_rgb = np.mean(top, axis=(0, 1))
        center_rgb = np.mean(center, axis=(0, 1))
        bottom_rgb = np.mean(bottom, axis=(0, 1))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        ax.text(10, 20, f'Top RGB: {top_rgb.astype(int)}', 
                color='white', fontsize=12, 
                bbox=dict(facecolor='black', alpha=0.5))
        ax.text(10, h // 3 + 20, f'Center RGB: {center_rgb.astype(int)}', 
                color='white', fontsize=12, 
                bbox=dict(facecolor='black', alpha=0.5))
        ax.text(10, 2 * h // 3 + 20, f'Bottom RGB: {bottom_rgb.astype(int)}', 
                color='white', fontsize=12, 
                bbox=dict(facecolor='black', alpha=0.5))
        ax.axhline(h // 3, color='yellow', linestyle='--', linewidth=1)
        ax.axhline(2 * h // 3, color='yellow', linestyle='--', linewidth=1)
        ax.axis('off')

        return fig


def predict_tss(storage_time, dafs, weight, avg_r, avg_g, avg_b, target="TSS"):
    """
    Predict Total Soluble Solids (TSS) using CatBoost model.
    
    Args:
        storage_time: Days in storage
        dafs: Days After Flowering
        weight: Weight in grams
        avg_r: Average red channel value
        avg_g: Average green channel value
        avg_b: Average blue channel value
        target: Target variable name for model file (default: "TSS")
    
    Returns:
        Predicted TSS value or None if model not found
    """
    volume = 250  # Assume volume (can be adjusted)
    w_c_ratio = weight / volume
    features = [[storage_time, dafs, weight, volume, w_c_ratio, avg_r, avg_g, avg_b]]
    
    model_path = MODEL_PATHS["ripeness"].format(target=target)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None

    model = CatBoostRegressor()
    model.load_model(model_path)
    return model.predict(features)[0]


def estimate_textural_attributes_from_tss(tss):
    """
    Estimate textural attributes from TSS value.
    
    Args:
        tss: Total Soluble Solids value
    
    Returns:
        Tuple of (peel_firmness, pulp_firmness, fruit_firmness)
    """
    peel_firmness = -0.55 * tss + 10.8
    pulp_firmness = -0.5 * tss + 10.0
    fruit_firmness = -0.52 * tss + 10.4
    return peel_firmness, pulp_firmness, fruit_firmness


def predict_sensory_from_tss(tss):
    """
    Predict sensory attributes from TSS value.
    
    Args:
        tss: Total Soluble Solids value
    
    Returns:
        Dictionary with sensory attribute predictions
    """
    pf, puf, ff = estimate_textural_attributes_from_tss(tss)
    sensory = {}
    sensory["Estimated Peel Firmness"] = round(pf, 2)
    sensory["Estimated Pulp Firmness"] = round(puf, 2)
    sensory["Estimated Fruit Firmness"] = round(ff, 2)
    sensory["Taste"] = round(16.844 * np.exp(-0.1024 * pf), 2)
    sensory["Appearance"] = round(47.479 * (pf ** -0.909), 2)
    sensory["Flavour"] = round(-2.0185 * np.log(ff) + 10.981, 2)
    sensory["Overall Acceptability"] = round(-0.0155 * pf**2 + 0.5671 * pf + 0.1008, 2)
    return sensory


def plot_radar_chart(scores):
    """
    Create and display a radar chart for sensory scores.
    
    Args:
        scores: Dictionary with attribute names as keys and scores as values
    """
    labels = list(scores.keys())
    values = list(scores.values())

    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='green', linewidth=2)
    ax.fill(angles, values, color='green', alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Sensory Quality Radar Chart", size=14, pad=20)
    plt.tight_layout()

    st.pyplot(fig)


def interpret_score(attr, value):
    """
    Interpret a sensory score and return a human-readable label.
    
    Args:
        attr: Attribute name (e.g., "Taste", "Flavour", "Appearance")
        value: Score value
    
    Returns:
        String with emoji and interpretation
    """
    if attr in ["Taste", "Flavour"]:
        if value < 5:
            return "😖 Poor"
        elif value < 8:
            return "🙂 Fair"
        elif value < 10:
            return "😋 Good"
        else:
            return "🤤 Excellent"

    if attr == "Appearance":
        if value < 5:
            return "😕 Dull"
        elif value < 10:
            return "😊 Nice"
        else:
            return "✨ Vibrant"

    if attr == "Overall Acceptability":
        if value < 1:
            return "❌ Low"
        elif value < 2:
            return "⚠️ Medium"
        else:
            return "✅ High"

    return ""  # For firmness, we just show numbers

