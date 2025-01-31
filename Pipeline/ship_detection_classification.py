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
from Pipeline.ship_image_scraper import scrape_image_selenium as scrape_image
import time

# Logging-Konfiguration: Meldungen ab "INFO" werden ausgegeben
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_unique_color():
    """
    Erzeugt eine zufällige BGR-Farbe als NumPy-Array.
    Wird fürs Einfärben von Segmentierungsbereichen benötigt.
    """
    return np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], dtype=np.uint8)

def save_image(image_np, path, filename):
    """
    Speichert das übergebene NumPy-Array (Bild) im angegebenen Pfad.
    Erstellt das Verzeichnis bei Bedarf.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = os.path.join(path, filename)
    cv2.imwrite(image_path, image_np)
    return image_path

def process_image(image, yolo_model, detection_labels, model_classification, device, process_id):
    """
    Führt sowohl YOLO-Segmentierung als auch Klassifizierung (auf erkannten Bbox-Ausschnitten) durch.
    Zeichnet Bounding Box und Label ins Bild und speichert das resultierende Bild ab.
    """
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

        # Prüfe Konfidenz und gewünschten Label-Typ
        if confidence > 0.68 and label in detection_labels:
            detected_boxes += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_image = image.crop((x1, y1, x2, y2))
            label_pred = classify_and_log(cropped_image, model_classification, device)

            # Info ins Bild einzeichnen
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
    image_path = save_image(image_np, "output", f"{process_id}_processed.jpg")

    return image_path

def classify_image(image, model_classification, device):
    """
    Konvertiert das gegebene PIL-Bild in einen Tensor und lässt
    ihn durch das Klassifikationsmodell laufen. Gibt die erkannte Klasse als int zurück.
    """
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
    """
    Ruft die Klassifizierungsfunktion auf, mappt die numerische Klasse auf
    einen (Schiffs-)Namen und protokolliert diesen.
    """
    label_pred = map_number_to_ship(classify_image(image, model_classification, device))
    logger.info(f"Schiffkategorie ist : {label_pred}")
    return label_pred

def process_single_image(process_id, yolo_model, detection_labels, model_classification, device):
    """
    Lädt das Bild von ShipSpotting.com via Selenium, führt YOLO und Klassifizierung durch
    und gibt den Pfad zum verarbeiteten Bild zurück.
    """
    image = scrape_image(process_id)
    if image is None:
        return None, None

    image_path = process_image(image, yolo_model, detection_labels, model_classification, device, process_id)
    return image_path

def main():
    """
    Hauptfunktion für die Bildverarbeitungspipeline:
    1. Gerät wählen (CPU/GPU)
    2. YOLO- und Klassifikationsmodell laden
    3. Liste von ShipSpotting-IDs verarbeiten
    4. Zwischen jedem Bild kurze Wartezeit
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Verwende Gerät: {device}")

    yolo_model = YOLO('/private/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/yolo_model/weights/best.pt')
    # Überprüfen Sie, ob YOLO das Gerät setzen kann
    if hasattr(yolo_model, 'to'):
        yolo_model.to(device)
    else:
        logger.warning("YOLO-Modell unterstützt die `.to(device)` Methode nicht.")

    # Klassifikationsmodell (ResNet50) konfigurieren
    num_classes = 9
    model_classification = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model_classification.fc.in_features
    model_classification.fc = nn.Linear(num_ftrs, num_classes)

    # Pfad zu den gespeicherten Gewichten
    model_save_path = "/private/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/category_model/best_model_trial_6.pth"

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
    ]

    for process_id in process_ids:
        logger.info(f"Verarbeite process_id: {process_id}")
        process_single_image(process_id, yolo_model, detection_labels, model_classification, device)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
