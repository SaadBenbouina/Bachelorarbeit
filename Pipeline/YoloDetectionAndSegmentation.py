import numpy as np
import random
import cv2
import torchvision.transforms as transforms
from ultralytics import YOLO
import torch
from torchvision import models
import torch.nn as nn
import os
import logging
from Pipeline.Mapper import map_number_to_ship
from Pipeline.ship_image_scraper import scrape_image_selenium as scrape_image  # Selenium-basierte Funktion importieren
import time

# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_unique_color():
    return np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], dtype=np.uint8)

def save_image(image_np, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = os.path.join(path, filename)
    cv2.imwrite(image_path, image_np)
    return image_path

def process_image(image, yolo_model, detection_labels, model_classification, device, process_id):
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    # Führe YOLO-Detektion und Segmentierung durch
    yolo_result = yolo_model.predict(image_np, task='segment')
    detected_boxes = 0
    boxes_data = []
    drawn_masks = []

    detections = yolo_result[0]

    for i, box in enumerate(detections.boxes):
        class_id = int(box.cls[0])
        label = yolo_model.names[class_id]
        confidence = box.conf[0]

        if confidence > 0.68 and label in detection_labels:
            detected_boxes += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_image = image.crop((x1, y1, x2, y2))
            label_pred = classify_and_log(cropped_image, model_classification, device)

            label_text = f'{label_pred} {confidence:.2f}'
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            boxes_data.append({
                'label': label_pred,
                'confidence': confidence,
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2
            })

            if detections.masks is not None:
                mask = detections.masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_resized = mask_resized.astype(bool)

                color = generate_unique_color()
                image_np[mask_resized] = (image_np[mask_resized] * 0.5 + color * 0.5).astype(np.uint8)

                drawn_masks.append((confidence, label_pred, mask_resized))

    if detected_boxes == 0:
        logger.warning("Keine Objekte in YOLO-Detektion gefunden.")
        return None, None

    # Speichere das verarbeitete Bild und die XML-Daten
    image_path = save_image(image_np, "output2", f"{process_id}_processed.jpg")

    return image_path

def classify_image(image, model_classification, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model_classification(image_tensor)
        _, preds = torch.max(outputs, 1)

    return preds.item()

def classify_and_log(image, model_classification, device):
    label_pred = map_number_to_ship(classify_image(image, model_classification, device))
    logger.info(f"Schiffkategorie ist : {label_pred}")
    return label_pred

def process_single_image(process_id, yolo_model, detection_labels, model_classification, device):
    image = scrape_image(process_id)
    if image is None:
        return None, None

    image_path = process_image(image, yolo_model, detection_labels, model_classification, device, process_id)
    return image_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Verwende Gerät: {device}")

    yolo_model = YOLO('/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/optuna_trial_1/weights/best.pt')
    # Überprüfen Sie, ob YOLO das Gerät setzen kann
    if hasattr(yolo_model, 'to'):
        yolo_model.to(device)
    else:
        logger.warning("YOLO-Modell unterstützt die `.to(device)` Methode nicht.")

    # Klassifikationsmodell laden
    num_classes = 9
    model_classification = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model_classification.fc.in_features
    model_classification.fc = nn.Linear(num_ftrs, num_classes)

    # Pfad zu den gespeicherten Gewichten
    model_save_path = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/trial_3/best_model_trial_3.pth"
    if not os.path.exists(model_save_path):
        logger.error(f"Klassifikationsmodell nicht gefunden am Pfad: {model_save_path}")
        return

    state_dict = torch.load(model_save_path, map_location=device)
    model_classification.load_state_dict(state_dict)
    model_classification.to(device)
    model_classification.eval()

    detection_labels = ["boat"]
    process_ids = [
        15001, 15002, 15003, 15004, 15005, 15006, 15007, 15008, 15009, 15010,
        15011, 15012, 15013, 15014, 15015, 15016, 15017, 15018, 15019, 15020,
        15021, 15022, 15023, 15024, 15025, 15026, 15027, 15028, 15029, 15030,
        15031, 15032, 15033, 15034, 15035, 15036, 15037, 15038, 15039, 15040,
        15041, 15042, 15043, 15044, 15045, 15046, 15047, 15048, 15049, 15050,
        15051, 15052, 15053, 15054, 15055, 15056, 15057, 15058, 15059, 15060,
        15061, 15062, 15063, 15064, 15065, 15066, 15067, 15068, 15069, 15070,
        15071, 15072, 15073, 15074, 15075, 15076, 15077, 15078, 15079, 15080,
        15081, 15082, 15083, 15084, 15085, 15086, 15087, 15088, 15089, 15090,
        15091, 15092, 15093, 15094, 15095, 15096, 15097, 15098, 15099, 15100,
        15101, 15102, 15103, 15104, 15105, 15106, 15107, 15108, 15109, 15110,
        15111, 15112, 15113, 15114, 15115, 15116, 15117, 15118, 15119, 15120,
        15121, 15122, 15123, 15124, 15125, 15126, 15127, 15128, 15129, 15130,
        15131, 15132, 15133, 15134, 15135, 15136, 15137, 15138, 15139, 15140,
        15141, 15142, 15143, 15144, 15145, 15146, 15147, 15148, 15149, 15150,
        15151, 15152, 15153, 15154, 15155, 15156, 15157, 15158, 15159, 15160,
        15161, 15162, 15163, 15164, 15165, 15166, 15167, 15168, 15169, 15170,
        15171, 15172, 15173, 15174, 15175, 15176, 15177, 15178, 15179, 15180,
        15181, 15182, 15183, 15184, 15185, 15186, 15187, 15188, 15189, 15190,
        15191, 15192, 15193, 15194, 15195, 15196, 15197, 15198, 15199, 15200,
        15201, 15202, 15203, 15204, 15205, 15206, 15207, 15208, 15209, 15210,
        15211, 15212, 15213, 15214, 15215, 15216, 15217, 15218, 15219, 15220,
        15221, 15222, 15223, 15224, 15225, 15226, 15227, 15228, 15229, 15230,
        15231, 15232, 15233, 15234, 15235, 15236, 15237, 15238, 15239, 15240,
        15241, 15242, 15243, 15244, 15245, 15246, 15247, 15248, 15249, 15250,
        15251, 15252, 15253, 15254, 15255, 15256, 15257, 15258, 15259, 15260,
        15261, 15262, 15263, 15264, 15265, 15266, 15267, 15268, 15269, 15270,
        15271, 15272, 15273, 15274, 15275, 15276, 15277, 15278, 15279, 15280,
        15281, 15282, 15283, 15284, 15285, 15286, 15287, 15288, 15289, 15290,
        15291, 15292, 15293, 15294, 15295, 15296, 15297, 15298, 15299, 15300,
        15301, 15302, 15303, 15304, 15305, 15306, 15307, 15308, 15309, 15310,
        15311, 15312, 15313, 15314, 15315, 15316, 15317, 15318, 15319, 15320,
        15321, 15322, 15323, 15324, 15325, 15326, 15327, 15328, 15329, 15330,
        15331, 15332, 15333, 15334, 15335, 15336, 15337, 15338, 15339, 15340,
        15341, 15342, 15343, 15344, 15345, 15346, 15347, 15348, 15349, 15350,
        15351, 15352, 15353, 15354, 15355, 15356, 15357, 15358, 15359, 15360,
        15361, 15362, 15363, 15364, 15365, 15366, 15367, 15368, 15369, 15370,
        15371, 15372, 15373, 15374, 15375, 15376, 15377, 15378, 15379, 15380,
        15381, 15382, 15383, 15384, 15385, 15386, 15387, 15388, 15389, 15390,
        15391, 15392, 15393, 15394, 15395, 15396, 15397, 15398, 15399, 15400,
        15401, 15402, 15403, 15404, 15405, 15406, 15407, 15408, 15409, 15410,
        15411, 15412, 15413, 15414, 15415, 15416, 15417, 15418, 15419, 15420,
        15421, 15422, 15423, 15424, 15425, 15426, 15427, 15428, 15429, 15430,
        15431, 15432, 15433, 15434, 15435, 15436, 15437, 15438, 15439, 15440,
        15441, 15442, 15443, 15444, 15445, 15446, 15447, 15448, 15449, 15450,
    ]

    for process_id in process_ids:
        logger.info(f"Verarbeite process_id: {process_id}")
        process_single_image(process_id, yolo_model, detection_labels, model_classification, device)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
