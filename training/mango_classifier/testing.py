import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import timm

# --- Load Model ---
model = timm.create_model('repvgg_a0', pretrained=False, num_classes=15)  # Change num_classes as needed
model.load_state_dict(torch.load(r"D:\CS\AI\PROJECT\ML_agri project\new_models\fruit_variety\repvgg_mango_classifier.pth", map_location='cpu'))
model.eval()

# --- Define Class Names ---
class_names = ['alphonsa', 'Ambika', 'Amrapali', 'Banganpalli', 'Chausa', 'Dasheri',
'Himsagar', 'Kesar', 'Langra', 'Malgova', 'Mallika', 'Neelam', 'Raspuri', 'totapuri',
'Vanraj']  # Modify based on your classes


transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means/stds
                             std=[0.229, 0.224, 0.225])
    ])


def circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:
        return 0
    return 4 * np.pi * (area / (perimeter * perimeter))


def detect_and_classify(image_path):
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Image not loaded. Check path: {image_path}")

    # Resize for processing consistency
    original = cv2.resize(original, (640, 480))
    output = original.copy()

    # Convert to HSV for color masking yellow-green hues (mango colors)
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    lower_yellow_green = np.array([20, 40, 40])
    upper_yellow_green = np.array([80, 255, 255])
    color_mask = cv2.inRange(hsv, lower_yellow_green, upper_yellow_green)

    # Grayscale for texture & edge detection
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Texture detection - Laplacian filter to detect textured mango skin
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    _, texture_mask = cv2.threshold(laplacian_abs, 20, 255, cv2.THRESH_BINARY)

    # Edge detection using Canny
    edges = cv2.Canny(gray, 50, 150)

    # Combine texture and edge detection masks (logical AND)
    texture_edge_mask = cv2.bitwise_and(texture_mask, edges)

    # Combine with color mask to remove non-mango objects
    combined_mask = cv2.bitwise_and(color_mask, texture_edge_mask)

    # Find contours on combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hough Circle Detection on grayscale for circular shapes
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=30, maxRadius=200)

    mango_candidates = []

    # Check contours for circularity > 0.6
    for cnt in contours:
        circ = circularity(cnt)
        if circ > 0.6:
            # Get bounding box around contour
            x, y, w, h = cv2.boundingRect(cnt)
            mango_candidates.append((x, y, w, h))

    # Also include Hough circles (if any)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x_c, y_c, r_c) in circles:
            mango_candidates.append((x_c - r_c, y_c - r_c, 2 * r_c, 2 * r_c))

    # Remove duplicates and keep candidates inside image bounds
    mango_candidates_filtered = []
    img_h, img_w = gray.shape
    for (x, y, w, h) in mango_candidates:
        x, y = max(x, 0), max(y, 0)
        w, h = min(w, img_w - x), min(h, img_h - y)
        if w > 20 and h > 20:  # ignore too small boxes
            mango_candidates_filtered.append((x, y, w, h))

    # Final: classify each candidate
    for (x, y, w, h) in mango_candidates_filtered:
        crop = original[y:y + h, x:x + w]
        # Convert crop to PIL image, apply model transforms
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            conf = conf.item()
            pred = pred.item()

        if conf > 0.56:
            label = f"{class_names[pred]}: {conf:.2f}"
            # Draw bounding box & label on output image
            center_x = x + w // 2
            center_y = y + h // 2
            radius = max(w, h) // 2
            cv2.circle(output, (center_x, center_y), radius, (0, 255, 0), 2)
            cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    # Show result
    cv2.imshow("Mango Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Run on your image
detect_and_classify(r"C:\Users\sabin\Downloads\images (7).jpeg")