import joblib
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# Load models


# Class names
class_names = {0: 'Undamaged', 1: 'Mild Damage', 2: 'Severe Damage'}
def preprocess_and_extract_features(images):
    features = []
    for img in images:
        resized = cv2.resize(img, (100, 100))  # Resize for uniformity
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # RGB → Grayscale

        # Noise Removal: Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thresholding: Grayscale → Binary
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

        # Extract RGB Mean Values
        avg_color_per_row = np.average(resized, axis=0)
        avg_colors = np.average(avg_color_per_row, axis=0)  # BGR
        b, g, r = avg_colors

        # Extract Size (Area) and Shape (Aspect Ratio)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = 0
        aspect_ratio = 0
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h if h != 0 else 0

        features.append([r, g, b, area, aspect_ratio])
    return np.array(features)

# Test function for new image
def classify_with_both_models(img_path, nb_model, rf_model):
    img = cv2.imread(img_path)
    if img is None:
        print("Error loading image.")
        return

    # Extract features from image
    features = preprocess_and_extract_features([img])  # Your earlier function

    # Predictions
    pred_nb = nb_model.predict(features)[0]
    pred_rf = rf_model.predict(features)[0]

    # Print results
    print(f"\n🟡 Naive Bayes Prediction: {class_names[pred_nb]}")
    print(f"🟢 Random Forest Prediction: {class_names[pred_rf]}")

    # Show image with predictions
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"NB: {class_names[pred_nb]} | RF: {class_names[pred_rf]}")
    plt.axis('off')
    plt.show()

    return pred_nb, pred_rf

def plot_prediction_comparison(pred_nb, pred_rf):
    predictions = [pred_nb, pred_rf]
    model_names = ['Naive Bayes', 'Random Forest']
    colors = ['orange', 'green']

    plt.bar(model_names, predictions, color=colors)
    plt.ylim(0, 2.5)
    plt.yticks([0, 1, 2], ['Undamaged', 'Mild Damage', 'Severe Damage'])
    plt.title('Model Prediction Comparison')
    plt.ylabel('Predicted Class')
    plt.show()
# Path to your new test image
test_image_path = input('Please insert image here!') # Change as needed

# Classify and plot
pred_nb, pred_rf = classify_with_both_models(test_image_path, nb_model, rf_model)
plot_prediction_comparison(pred_nb, pred_rf)

