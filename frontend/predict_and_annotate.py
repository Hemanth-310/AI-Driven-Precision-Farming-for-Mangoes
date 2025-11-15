import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import timm

# Load model
model = timm.create_model('repvgg_a0', pretrained=False, num_classes=15)
model.load_state_dict(torch.load(r"D:\CS\AI\PROJECT\ML_agri project\new_models\fruit_variety\repvgg_mango_classifier.pth", map_location='cpu'))
model.eval()

class_names = ['alphonsa', 'Ambika', 'Amrapali', 'Banganpalli', 'Chausa', 'Dasheri',
               'Himsagar', 'Kesar', 'Langra', 'Malgova', 'Mallika', 'Neelam', 'Raspuri', 'totapuri', 'Vanraj']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:
        return 0
    return 4 * np.pi * (area / (perimeter * perimeter))


def predict_and_annotate(image_path):
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError("Failed to load image.")

    original = cv2.resize(original, (640, 480))
    output = original.copy()

    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    lower_yellow_green = np.array([20, 40, 40])
    upper_yellow_green = np.array([80, 255, 255])
    color_mask = cv2.inRange(hsv, lower_yellow_green, upper_yellow_green)

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    _, texture_mask = cv2.threshold(laplacian_abs, 20, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 50, 150)
    texture_edge_mask = cv2.bitwise_and(texture_mask, edges)
    combined_mask = cv2.bitwise_and(color_mask, texture_edge_mask)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=30, maxRadius=200)

    mango_candidates = []
    for cnt in contours:
        circ = circularity(cnt)
        if circ > 0.6:
            x, y, w, h = cv2.boundingRect(cnt)
            mango_candidates.append((x, y, w, h))

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x_c, y_c, r_c) in circles:
            mango_candidates.append((x_c - r_c, y_c - r_c, 2 * r_c, 2 * r_c))

    img_h, img_w = gray.shape
    mango_candidates_filtered = []
    for (x, y, w, h) in mango_candidates:
        x, y = max(x, 0), max(y, 0)
        w, h = min(w, img_w - x), min(h, img_h - y)
        if w > 20 and h > 20:
            mango_candidates_filtered.append((x, y, w, h))

    predicted_labels = []

    for (x, y, w, h) in mango_candidates_filtered:
        crop = original[y:y + h, x:x + w]
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            conf = conf.item()
            pred = pred.item()

        if conf > 0.50:
            label = f"{class_names[pred]}: {conf:.2f}"
            predicted_labels.append(label)
            center_x = x + w // 2
            center_y = y + h // 2
            radius = max(w, h) // 2
            #cv2.circle(output, (center_x, center_y), radius, (0, 255, 0), 2)
            cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Convert annotated image for display
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    pil_output = Image.fromarray(output_rgb)

    return pil_output, predicted_labels
