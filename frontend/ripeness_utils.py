'''
def get_cultivar_from_state(state):
    state_cultivars = {
        'maharashtra': 'Alphonso',
        'andhra pradesh': 'Banganapalli',
        'tamil nadu': 'Neelam',
        'punjab': 'chausa',
        'uttar pradesh': 'Dashehri',
        'bihar': 'Maldah',
        'gujarat': 'Kesar'
    }

    # Convert the state to lowercase to handle case-insensitive comparison
    state = state.lower()

    # Return the cultivar based on the state
    return state_cultivars.get(state, None)


def get_sensory_predictions(state, ripeness_stage, x_value):
    """Function to apply the models based on state, ripeness stage, and cultivar."""
    cultivar = get_cultivar_from_state(state)

    if cultivar == "Alphonso":
        # Equations for Alphonso cultivar
        if ripeness_stage == "mid-ripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)
        elif ripeness_stage == "ripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)
        elif ripeness_stage == "unripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)

        return {
            "peel_firmness": peel_firmness,
            "flavor": flavor,
            "pulp_firmness": pulp_firmness,
            "fruit_firmness": fruit_firmness,
            "overall": overall
        }

    elif cultivar == "Banganapalli":
        if ripeness_stage == "mid-ripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)
        elif ripeness_stage == "ripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)
        elif ripeness_stage == "unripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)
        return {
            "peel_firmness": peel_firmness,
            "flavor": flavor,
            "pulp_firmness": pulp_firmness,
            "fruit_firmness": fruit_firmness,
            "overall": overall}


    elif cultivar == "Maldah":
        if ripeness_stage == "mid-ripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)
        elif ripeness_stage == "ripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)
        elif ripeness_stage == "unripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)

        return {
            "peel_firmness": peel_firmness,
            "flavor": flavor,
            "pulp_firmness": pulp_firmness,
            "fruit_firmness": fruit_firmness,
            "overall": overall
        }

    elif cultivar == "Chausa":
        if ripeness_stage == "mid-ripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)
        elif ripeness_stage == "ripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)
        elif ripeness_stage == "unripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)

        return {
            "peel_firmness": peel_firmness,
            "flavor": flavor,
            "pulp_firmness": pulp_firmness,
            "fruit_firmness": fruit_firmness,
            "overall": overall
        }

    elif cultivar == "Neelam":
        if ripeness_stage == "mid-ripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)
        elif ripeness_stage == "ripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)
        elif ripeness_stage == "unripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)

        return {
            "peel_firmness": peel_firmness,
            "flavor": flavor,
            "pulp_firmness": pulp_firmness,
            "fruit_firmness": fruit_firmness,
            "overall": overall
        }

    elif cultivar == "Kesar":
        if ripeness_stage == "mid-ripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)
        elif ripeness_stage == "ripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)
        elif ripeness_stage == "unripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)

        return {
            "peel_firmness": peel_firmness,
            "flavor": flavor,
            "pulp_firmness": pulp_firmness,
            "fruit_firmness": fruit_firmness,
            "overall": overall
        }

    elif cultivar == "Dashehri":
        if ripeness_stage == "mid-ripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)
        elif ripeness_stage == "ripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)
        elif ripeness_stage == "unripe":
            peel_firmness = 9.6735 * (10 ** (-0.0309 * x_value))
            flavor = 0.0747 * x_value ** 2 + 0.2309 * x_value + 6.0365
            pulp_firmness = 8.9195 * (10 ** (-0.0743 * x_value))
            fruit_firmness = 8.9168 * x_value ** (-0.1424)
            overall = 9.389 * x_value ** (-0.1614)

        return {
            "peel_firmness": peel_firmness,
            "flavor": flavor,
            "pulp_firmness": pulp_firmness,
            "fruit_firmness": fruit_firmness,
            "overall": overall
        }


def get_ripeness_stage_value(ripeness_stage):
    # Mapping of ripeness stages to numeric values
    ripeness_stage_mapping = {
        "ripe": 1,
        "mid-ripe": 2,
        "unripe": 3
    }

    # Return the numeric value corresponding to the ripeness stage
    return ripeness_stage_mapping.get(ripeness_stage.lower(), None)


def classify_sensory_properties(peel_firmness, flavor, pulp_firmness, fruit_firmness, overall):
    # Classifying the peel firmness based on actual values
    if peel_firmness < 8:
        peel_firmness_statement = "The peel is firm, which could indicate the mango is slightly unripe,indicating it strength for transport."
    elif peel_firmness < 9:
        peel_firmness_statement = "The peel is slightly firm. This is a good stage for handling and packaging."
    elif peel_firmness < 10:
        peel_firmness_statement = "The peel is quite soft. You may want to handle it gently to avoid damage."
    else:
        peel_firmness_statement = "The peel is very firm, which could indicate the mango is slightly overripe, but it's still in good condition for sale."

    # Classifying the flavor based on actual values
    if flavor < 6.5:
        flavor_statement = "The flavor is outstanding, with a very sweet and complex profile. The mango is at its peak."
    elif flavor < 7:
        flavor_statement = "The flavor is good, with a noticeable sweetness starting to emerge. It's nearing its peak.""The flavor is slightly developed but still needs more time to reach its peak sweetness."
    elif flavor < 7.5:
        flavor_statement = "The flavor is still developing and might taste bland. It’s better to wait a bit before selling."
    else:
        flavor_statement = " "

    # Classifying pulp firmness based on actual values
    if pulp_firmness < 5.5:
        pulp_firmness_statement = "The pulp is firm, which may indicate the mango is still unripe or just slightly ripe."
    elif pulp_firmness < 6.5:
        pulp_firmness_statement = "The pulp is soft, typical for a mango nearing full ripeness."
    elif pulp_firmness < 8:
        pulp_firmness_statement = "The pulp is very soft, which indicates that the mango is ripe."
    else:
        pulp_firmness_statement = " "

    # Classifying fruit firmness based on actual values
    if fruit_firmness < 8:
        fruit_firmness_statement = "The fruit is firm, suggesting that the mango is unripe or just reaching the ripe stage."
    elif fruit_firmness < 8.5:
        fruit_firmness_statement = "The fruit is moderately firm, indicating it is near ripe."
    elif fruit_firmness < 9:
        fruit_firmness_statement = "The fruit is soft, which is typical of a ripe mango."
    else:
        fruit_firmness_statement = " "

    # Classifying the overall quality based on actual values
    if overall < 8:
        overall_statement = "The overall quality is fair. The mango is likely unripe .Make pickles!"
    elif overall < 9:
        overall_statement = "The overall quality is fair. The mango may still require some ripening to reach peak flavor."
    elif overall < 10:
        overall_statement = "The overall quality is very good. The mango is at a good stage of ripeness for flavor and texture."
    else:
        overall_statement = "The overall quality is excellent. The mango is at peak ripeness, offering the best flavor and texture."

    # Returning the complete classification statements
    return {
        "peel_firmness": peel_firmness_statement,
        "flavor": flavor_statement,
        "pulp_firmness": pulp_firmness_statement,
        "fruit_firmness": fruit_firmness_statement,
        "overall": overall_statement
    }


# Main function
import cv2
import matplotlib.pyplot as plt


def main():
    state = input(
        'Enter your state \n(Maharashtra, Andhra Pradesh, Tamil Nadu, Punjab, Uttar Pradesh, Bihar, Gujarat): ')
    harvesting_period = input(
        'Enter harvesting period \n(First-March to April, Second-April to May, Third-May to June): ')
    image_path = input("Enter the path of the mango image: ")
    try:
        storage_time = float(input("Enter Storage Time: "))
        daf = float(input("Enter DAF: "))
        weight = float(input("Enter Mango Weight: "))
    except ValueError:
        print("Invalid input. Please enter numeric values for Storage Time, DAF, and Weight.")
        return

    # Load model
    try:
        ripeness_model = load_model('mango_ripeness_model.h5')
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    # Extract RGB values from the image
    rgb_values = extract_rgb_values(image_path)

    if rgb_values:
        top_rgb, center_rgb, bottom_rgb = rgb_values
        print(f"Top RGB: {top_rgb}")
        print(f"Center RGB: {center_rgb}")
        print(f"Bottom RGB: {bottom_rgb}")

        # Classify ripeness stage based on RGB values
        ripeness_stage_rgb = classify_ripeness_stage(rgb_values)
        print(f"RGB-based ripeness stage: {ripeness_stage_rgb}")

        # Map the ripeness stage to a numeric value
        ripeness_value = get_ripeness_stage_value(ripeness_stage_rgb)
        print(f"Numeric ripeness value: {ripeness_value}")

        # Get cultivar based on the state
        cultivar = get_cultivar_from_state(state)

        if cultivar:
            # Calculate x_value from ripeness stage value (You can modify this logic if needed)
            if ripeness_value == 1:
                x_value = 1  # for ripe
            elif ripeness_value == 2:
                x_value = 2  # for mid-ripe
            elif ripeness_value == 3:
                x_value = 3  # for unripe
            else:
                print("Error: Invalid ripeness stage.")
                return

            # Get sensory predictions based on the state, ripeness stage, and x_value
            sensory_predictions = get_sensory_predictions(state, ripeness_stage_rgb, x_value)

            # Check if sensory predictions are None
            if sensory_predictions is not None:
                # Classify sensory properties and print the statements
                sensory_properties = classify_sensory_properties(
                    sensory_predictions['peel_firmness'],
                    sensory_predictions['flavor'],
                    sensory_predictions['pulp_firmness'],
                    sensory_predictions['fruit_firmness'],
                    sensory_predictions['overall']
                )
                print(sensory_properties["peel_firmness"])
                print(sensory_properties["flavor"])
                print(sensory_properties["pulp_firmness"])
                print(sensory_properties["fruit_firmness"])
                print(sensory_properties["overall"])
            else:
                print("Error: Sensory predictions could not be generated.")

            # Visualize the mango image with ripeness stage as title
            img = cv2.imread(image_path)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"Mango - {ripeness_stage_rgb} stage")
            plt.show()
        else:
            print("sorry, we don't provide service for your location")
    else:
        print("Could not extract RGB values from the image.")


# Run the main function
main()
'''
'''
#NEW CODE

import cv2
import numpy as np
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import os

# === RGB Processing Class ===
class ImageProcessing:
    def __init__(self, image_path):
        self.image_path = image_path

    def extract_rgb_values(self):
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

        return get_avg(top), get_avg(center), get_avg(bottom)

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

    def get_avg_rgb(self):
        top_rgb, center_rgb, bottom_rgb = self.extract_rgb_values()
        if top_rgb is None:
            return None
        avg_r = (top_rgb[0] + center_rgb[0] + bottom_rgb[0]) / 3
        avg_g = (top_rgb[1] + center_rgb[1] + bottom_rgb[1]) / 3
        avg_b = (top_rgb[2] + center_rgb[2] + bottom_rgb[2]) / 3
        return avg_r, avg_g, avg_b

# === TSS Prediction using CatBoost ===
def predict_tss(storage_time, dafs, weight, avg_r, avg_g, avg_b):
    volume = 250  # Assume volume (can be adjusted)
    w_c_ratio = weight / volume

    features = [[storage_time, dafs, weight, volume, w_c_ratio, avg_r, avg_g, avg_b]]

    model_path = r"D:\CS\AI\PROJECT\ML_agri project\new_models\ripeness\catboost_{target}.cbm"
    if not os.path.exists(model_path):
        print("❌ Model not found:", model_path)
        return None

    model = CatBoostRegressor()
    model.load_model(model_path)
    return model.predict(features)[0]

# === Sensory Prediction from Paper Equations ===
def predict_sensory(peel_firmness, fruit_firmness, pulp_firmness, pulp_toughness):
    results = {}

    results['Taste'] = 16.844 * np.exp(-0.1024 * peel_firmness)  # AM Cultivar
    results['Appearance'] = 47.479 * (peel_firmness ** -0.909)   # KM Cultivar
    results['Flavour'] = -2.0185 * np.log(fruit_firmness) + 10.981  # MB Cultivar
    results['Overall Acceptability'] = -0.0155 * peel_firmness**2 + 0.5671 * peel_firmness + 0.1008  # BA

    return results

# === Main Execution ===
if __name__ == "__main__":
    print("\n🍋 Mango Ripeness Prediction System")
    image_path = input("📷 Enter path to mango image: ")
    storage_time = float(input("📅 Storage Time (days): "))
    dafs = float(input("🌱 Days After Flowering (DAFS): "))
    weight = float(input("⚖️  Weight (grams): "))

    peel_firmness = float(input("🔬 Peel Firmness (e.g., 4.2): "))
    fruit_firmness = float(input("🔬 Fruit Firmness: "))
    pulp_firmness = float(input("🔬 Pulp Firmness: "))
    pulp_toughness = float(input("🔬 Pulp Toughness: "))

    processor = ImageProcessing(image_path)

    # RGB-based interpretation
    rgb_stage = processor.classify_rgb_stage()
    avg_r, avg_g, avg_b = processor.get_avg_rgb()

    # Predict TSS
    predicted_tss = predict_tss(storage_time, dafs, weight, avg_r, avg_g, avg_b)

    # Classify ripeness from TSS
    if predicted_tss < 8:
        tss_stage = "Unripe"
    elif predicted_tss < 11:
        tss_stage = "Mid-Ripe"
    else:
        tss_stage = "Ripe"

    # Predict sensory parameters
    sensory = predict_sensory(peel_firmness, fruit_firmness, pulp_firmness, pulp_toughness)

    # === OUTPUT ===
    print("\n🎯 Final Ripeness Assessment")
    print(f"📸 RGB-Based Stage           : {rgb_stage}")
    print(f"🧪 Predicted TSS             : {predicted_tss:.2f} °Brix")
    print(f"🍃 TSS-Based Ripeness        : {tss_stage}")

    print("\n🌟 Estimated Sensory Attributes:")
    for key, val in sensory.items():
        print(f"  {key:<22}: {val:.2f}")
'''

import cv2
import numpy as np
from catboost import CatBoostRegressor
import os
import matplotlib.pyplot as plt
# === RGB Processing Class ===
class ImageProcessing:
    def __init__(self, image_path):
        self.image_path = image_path

    def extract_rgb_values(self):
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
        top, center, bottom, top_rgb, center_rgb, bottom_rgb = self.extract_rgb_values()
        if top_rgb is None:
            return None
        avg_r = (top_rgb[0] + center_rgb[0] + bottom_rgb[0]) / 3
        avg_g = (top_rgb[1] + center_rgb[1] + bottom_rgb[1]) / 3
        avg_b = (top_rgb[2] + center_rgb[2] + bottom_rgb[2]) / 3
        return avg_r, avg_g, avg_b

    def plot_rgb_with_image(self):
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
        ax.text(10, 20, f'Top RGB: {top_rgb.astype(int)}', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
        ax.text(10, h // 3 + 20, f'Center RGB: {center_rgb.astype(int)}', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
        ax.text(10, 2 * h // 3 + 20, f'Bottom RGB: {bottom_rgb.astype(int)}', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
        ax.axhline(h // 3, color='yellow', linestyle='--', linewidth=1)
        ax.axhline(2 * h // 3, color='yellow', linestyle='--', linewidth=1)
        ax.axis('off')

        return fig  # ✅ Return the figure to render in Streamlit

# === TSS Prediction ===
def predict_tss(storage_time, dafs, weight, avg_r, avg_g, avg_b):
    volume = 250
    w_c_ratio = weight / volume
    features = [[storage_time, dafs, weight, volume, w_c_ratio, avg_r, avg_g, avg_b]]
    model_path = r"D:\CS\AI\PROJECT\ML_agri project\new_models\ripeness\catboost_{target}.cbm"

    if not os.path.exists(model_path):
        print("❌ Model not found:", model_path)
        return None

    model = CatBoostRegressor()
    model.load_model(model_path)
    return model.predict(features)[0]

# === Estimate Textural Attributes from TSS ===
def estimate_textural_attributes_from_tss(tss):
    peel_firmness = -0.55 * tss + 10.8
    pulp_firmness = -0.5 * tss + 10.0
    fruit_firmness = -0.52 * tss + 10.4
    return peel_firmness, pulp_firmness, fruit_firmness

# === Predict Sensory Using Estimated Texture ===
def predict_sensory_from_tss(tss):
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
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st  # ✅ ADD this

def plot_radar_chart(scores):
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

    st.pyplot(fig)  # ✅ SHOW THE PLOT IN STREAMLIT



def interpret_score(attr, value):
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


# === Main Execution ===
if __name__ == "__main__":
    print("\n🍋 Mango Ripeness Prediction System")
    image_path = input("📷 Enter path to mango image: ")
    storage_time = float(input("📅 Storage Time (days): "))
    dafs = float(input("🌱 Days After Flowering (DAFS): "))
    weight = float(input("⚖️  Weight (grams): "))

    processor = ImageProcessing(image_path)
    rgb_stage = processor.classify_rgb_stage()
    avg_r, avg_g, avg_b = processor.get_avg_rgb()

    predicted_tss = predict_tss(storage_time, dafs, weight, avg_r, avg_g, avg_b)

    if predicted_tss is None:
        exit("🚫 Could not compute TSS.")

    if predicted_tss < 8:
        tss_stage = "Unripe"
    elif predicted_tss < 11:
        tss_stage = "Mid-Ripe"
    else:
        tss_stage = "Ripe"

    print("\n🎯 Final Ripeness Assessment")
    print(f"📸 RGB-Based Stage           : {rgb_stage}")
    print(f"🧪 Predicted TSS             : {predicted_tss:.2f} °Brix")
    print(f"🍃 TSS-Based Ripeness        : {tss_stage}")

    sensory_result = predict_sensory_from_tss(predicted_tss)

    print("\n🌟 Estimated Textural & Sensory Attributes:")
    radar_scores = {}

    for k, v in sensory_result.items():
        if k in ["Taste", "Flavour", "Appearance"]:
            label = interpret_score(k, v)
            print(f"  {k:<25}: {v:.2f} ({label})")
            radar_scores[k] = v
        else:
            print(f"  {k:<25}: {v:.2f}")  # For firmness

    # 📊 Show radar plot
    plot_radar_chart(radar_scores)
    # Plot RGB values on the image
    processor.plot_rgb_with_image()



