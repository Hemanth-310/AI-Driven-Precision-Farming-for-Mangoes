import cv2
import numpy as np

'''
class ImageProcessing:
    def __init__(self, image_path):
        self.image_path = image_path

    def extract_rgb_values(self):
        img = cv2.imread(self.image_path)
        if img is None:
            print("Error loading image")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, _ = img.shape
        top_region = img[0:int(height / 3), 0:width]
        center_region = img[int(height / 3):int(2 * height / 3), 0:width]
        bottom_region = img[int(2 * height / 3):height, 0:width]

        def get_average_rgb(region):
            return np.mean(region, axis=(0, 1))

        top_rgb = get_average_rgb(top_region)
        center_rgb = get_average_rgb(center_region)
        bottom_rgb = get_average_rgb(bottom_region)

        return top_rgb, center_rgb, bottom_rgb

    def classify_ripeness_stage(self):
        rgb_values = self.extract_rgb_values()
        if rgb_values is None:
            return None

        top_rgb, center_rgb, bottom_rgb = rgb_values

        avg_top_r = np.mean(top_rgb[0])
        avg_top_g = np.mean(top_rgb[1])
        avg_top_b = np.mean(top_rgb[2])

        avg_bottom_r = np.mean(bottom_rgb[0])
        avg_bottom_g = np.mean(bottom_rgb[1])
        avg_bottom_b = np.mean(bottom_rgb[2])

        # Check if the mango is ripe (red, orange, or yellow tones dominate)
        if (avg_top_r > avg_top_g and avg_top_r > avg_top_b) or (
                avg_bottom_r > avg_bottom_g and avg_bottom_r > avg_bottom_b):
            return "ripe"
        # Mid-ripe (green on top and yellow or orange on the bottom, or vice versa)
        elif (avg_top_g > avg_top_r and avg_bottom_r > avg_bottom_g) or (
                avg_top_r > avg_top_g and avg_bottom_g > avg_bottom_r):
            return "mid-ripe"
        # Unripe (mostly green)
        else:
            return "unripe"
'''
import cv2
import numpy as np
from catboost import CatBoostRegressor
import joblib  # for model loading
import os

# === RGB Image Class ===
class ImageProcessing:
    def __init__(self, image_path):
        self.image_path = image_path

    def extract_rgb_values(self):
        img = cv2.imread(self.image_path)
        if img is None:
            print("Error loading image.")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, _ = img.shape
        top = img[0:int(height / 3), :]
        center = img[int(height / 3):int(2 * height / 3), :]
        bottom = img[int(2 * height / 3):, :]

        def get_avg_rgb(region):
            return np.mean(region, axis=(0, 1))

        return get_avg_rgb(top), get_avg_rgb(center), get_avg_rgb(bottom)

    def classify_rgb_stage(self):
        top_rgb, center_rgb, bottom_rgb = self.extract_rgb_values()
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

    def get_avg_rgb_features(self):
        top_rgb, center_rgb, bottom_rgb = self.extract_rgb_values()
        if top_rgb is None:
            return None
        avg_r = (top_rgb[0] + center_rgb[0] + bottom_rgb[0]) / 3
        avg_g = (top_rgb[1] + center_rgb[1] + bottom_rgb[1]) / 3
        avg_b = (top_rgb[2] + center_rgb[2] + bottom_rgb[2]) / 3
        return avg_r, avg_g, avg_b

# === Prediction Wrapper ===
def predict_tss(storage_time, dafs, weight, avg_r, avg_g, avg_b):
    # Estimate volume from weight (e.g., assume density or fixed volume)
    volume = 250  # dummy volume in cm³
    w_c_ratio = weight / volume

    features = [[
        storage_time, dafs, weight, volume, w_c_ratio, avg_r, avg_g, avg_b
    ]]

    model_path = r"D:\CS\AI\PROJECT\ML_agri project\new_models\ripeness\catboost_{target}.cbm"
    if not os.path.exists(model_path):
        print("❌ Model not found:", model_path)
        return None

    model = CatBoostRegressor()
    model.load_model(model_path)
    prediction = model.predict(features)[0]
    return prediction

# === USER INPUT + RUN ===
if __name__ == "__main__":
    image_path = input(r"📷 Enter path to mango image: ")
    storage_time = float(input("📅 Enter Storage Time (days): "))
    dafs = float(input("🌱 Enter Days After Flowering (DAFS): "))
    weight = float(input("⚖️  Enter Weight (grams): "))

    processor = ImageProcessing(image_path)

    # RGB stage classification
    rgb_stage = processor.classify_rgb_stage()
    avg_r, avg_g, avg_b = processor.get_avg_rgb_features()

    # Predict TSS
    predicted_tss = predict_tss(storage_time, dafs, weight, avg_r, avg_g, avg_b)

    # Final output
    print("\n📊 Mango Ripeness Estimation:")
    print(f"   RRipeness level     : {rgb_stage}")

    # Optional: Interpret stage by TSS
    if predicted_tss >= 11:
        stage = "Ripe"
    elif 8 <= predicted_tss < 11:
        stage = "Mid-Ripe"
    else:
        stage = "Unripe"

    print(f"   TSS-based Stage      : {stage}")
import matplotlib.pyplot as plt

def visualize_rgb_classification(image_path, top_rgb, center_rgb, bottom_rgb, label):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # Show original image with overlays
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_rgb)
    ax.set_title(f"RGB-based Ripeness Stage: {label}", fontsize=14)
    ax.axis('off')

    # Annotate RGB values on regions
    ax.text(w // 2, h * 0.1, f"Top RGB:    {np.round(top_rgb).astype(int)}", color='white', fontsize=10, ha='center')
    ax.text(w // 2, h * 0.5, f"Center RGB: {np.round(center_rgb).astype(int)}", color='white', fontsize=10, ha='center')
    ax.text(w // 2, h * 0.9, f"Bottom RGB: {np.round(bottom_rgb).astype(int)}", color='white', fontsize=10, ha='center')

    plt.tight_layout()
    plt.show()

def predict_tss(storage_time, dafs, weight, avg_r, avg_g, avg_b):
    # Estimate volume from weight (e.g., assume density or fixed volume)
    volume = 250  # dummy volume in cm³
    w_c_ratio = weight / volume

    features = [[
        storage_time, dafs, weight, volume, w_c_ratio, avg_r, avg_g, avg_b
    ]]

    model_path = "saved_models/catboost_TSS.cbm"
    if not os.path.exists(model_path):
        print("❌ Model not found:", model_path)
        return None

    model = CatBoostRegressor()
    model.load_model(model_path)
    prediction = model.predict(features)[0]
    return prediction

# === USER INPUT + RUN ===
if __name__ == "__main__":
    image_path = input("📷 Enter path to mango image: ")
    storage_time = float(input("📅 Enter Storage Time (days): "))
    dafs = float(input("🌱 Enter Days After Flowering (DAFS): "))
    weight = float(input("⚖️  Enter Weight (grams): "))

    processor = ImageProcessing(image_path)

    # RGB stage classification
    rgb_stage = processor.classify_rgb_stage()
    top_rgb, center_rgb, bottom_rgb = processor.extract_rgb_values()
    avg_r, avg_g, avg_b = processor.get_avg_rgb_features()
    # === Show RGB Visualization ===
    visualize_rgb_classification(image_path, top_rgb, center_rgb, bottom_rgb, rgb_stage)
    # Predict TSS
    predicted_tss = predict_tss(storage_time, dafs, weight, avg_r, avg_g, avg_b)

    print("\n📊 Mango Ripeness Estimation:")
    print(f"   RGB-based Stage      : {rgb_stage}")

    if predicted_tss > 11 or predicted_tss == 11:
        stage = "Ripe"
    elif 8 < predicted_tss < 11:
        stage = "Mid-Ripe"
    else:
        stage = "Unripe"
    print(f"   TSS-based Stage      : {stage}")

