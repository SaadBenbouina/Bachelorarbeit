import numpy as np
import xml.etree.ElementTree as ET
import pycocotools.mask as mask_util
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
from Pipeline.ship_image_scraper import scrape_image

# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_unique_color():
    """
    Generiert eine eindeutige Farbe für jedes Objekt.
    """
    return np.array([random.randint(0, 255), random.randint(
        0, 255), random.randint(0, 255)], dtype=np.uint8)

def mask_to_rle(mask):
    """
    Konvertiert eine Maske in das RLE-Format.
    """
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle

def save_image(image_np, path, filename):
    """
    Speichert das verarbeitete Bild im angegebenen Pfad.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = os.path.join(path, filename)
    cv2.imwrite(image_path, image_np)  # Speichere das Bild
    return image_path

def process_image(image, yolo_model, detection_labels, photo_label, process_id):
    """
    Verarbeitet ein einzelnes Bild für Detektion, Segmentierung und Klassifikation.
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

        # Berücksichtige nur relevante Labels und genügend hohe Confidence
        if confidence > 0.45 and label in detection_labels:
            detected_boxes += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label_text = f'{label} {confidence:.2f}'
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            boxes_data.append({
                'label': label,
                'confidence': confidence,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            })

            # Wenn eine Maske vorhanden ist, zeichnen Sie sie
            if detections.masks is not None:
                # Holen Sie sich die Maske und skalieren Sie sie zurück auf die Originalgröße
                mask = detections.masks.data[i].cpu().numpy()
                # Masken haben die Größe des Modells, skalieren Sie sie zurück auf die Bildgröße
                mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_resized = mask_resized.astype(bool)

                # Zeichnen Sie die Maske auf das Originalbild
                color = generate_unique_color()
                image_np[mask_resized] = (image_np[mask_resized] * 0.5 + color * 0.5).astype(np.uint8)

                # Maske zu den gezeichneten Masken hinzufügen
                drawn_masks.append((confidence, label, mask_resized))

    if detected_boxes == 0:
        logger.warning("Keine Objekte in YOLO-Detektion gefunden.")
        return None, None

    # Speichere das verarbeitete Bild
    image_path = save_image(image_np, "output2",
                            f"{process_id}_processed.jpg")

    # Erstelle XML-Struktur zum Speichern der Bildmetadaten
    image_metadata = ET.Element("image", id=str(process_id), category=f"{photo_label}",
                                width=str(width), height=str(height))

    # Speichere die Bounding-Boxen im XML
    for box in boxes_data:
        box_element = ET.SubElement(
            image_metadata, "Bounding_box", label=box['label'], confidence=f"{box['confidence']:.2f}")
        box_element.set("x1", str(box['x1']))
        box_element.set("y1", str(box['y1']))
        box_element.set("x2", str(box['x2']))
        box_element.set("y2", str(box['y2']))

    # Speichere die gezeichneten Masken im XML
    for confidence, label, mask in drawn_masks:
        rle = mask_to_rle(mask)  # Konvertiere die Maske in RLE
        mask_element = ET.SubElement(
            image_metadata, "mask", label=label, source="YOLOv8")
        rle_str = ', '.join(str(count) for count in rle["counts"])
        mask_element.set("rle", rle_str)
        mask_element.set("confidence", f"{confidence:.2f}")

    return image_metadata, image_path

def classify_image(image, model_classification, device):
    """
    Klassifiziert ein Bild mit dem gegebenen Klassifikationsmodell.
    """
    # Definieren der Transformationen
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Wenden der Transformationen auf das Bild an
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model_classification(image_tensor)
        _, preds = torch.max(outputs, 1)

    return preds.item()


def save_xml(xml_data, file_name="output2.xml", path=""):
    """
    Speichert XML-Daten mit Detektions-, Segmentierungs- und Klassifikationsdaten.
    """
    # Wenn der Pfad nicht leer ist, stelle sicher, dass das Verzeichnis existiert
    if path:
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, file_name)
    else:
        file_path = file_name  # Speichere im aktuellen Verzeichnis, wenn kein Pfad angegeben ist

    # Speichere die XML-Daten im angegebenen Pfad und Dateinamen
    tree = ET.ElementTree(xml_data)
    # Einrücken der XML-Struktur für eine schönere Darstellung
    ET.indent(tree, space="  ", level=0)  # Füge Einrückungen und Zeilenumbrüche hinzu
    tree.write(file_path)
    return file_path

def classify_and_log(image, model_classification, device):
    label_pred = map_number_to_ship(classify_image(image, model_classification, device))
    logger.info(f"Schiffkategorie ist : {label_pred}")
    return label_pred

def process_single_image(process_id, yolo_model, detection_labels, model_classification, device):
    """
    Wrapper-Funktion zum Verarbeiten eines einzelnen Bildes anhand seiner process_id.
    """
    # Bild von der Webseite herunterladen
    image = scrape_image(process_id)
    if image is None:
        return None, None

    # Klassifikation durchführen und loggen
    label_pred = classify_and_log(image, model_classification, device)

    # Segmentierung und Bounding-Boxen erstellen
    xml_data, image_path = process_image(image, yolo_model, detection_labels, label_pred, process_id)
    return xml_data, image_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model = YOLO('yolov8x-seg.pt')
    yolo_model.to(device)

    # Klassifikationsmodell laden
    num_classes = 16
    model_classification = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model_classification.fc.in_features
    model_classification.fc = nn.Linear(num_ftrs, num_classes)

    # Pfad zu den gespeicherten Gewichten
    model_save_path = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/ship_classification_resnet50.pth"
    state_dict = torch.load(model_save_path, map_location=device)
    model_classification.load_state_dict(state_dict)
    model_classification.to(device)
    model_classification.eval()  # Setzt das Modell in den Evaluationsmodus

    detection_labels = ["boat"]
    process_ids = [76844, 96844, 126744, 154644, 154859, 933373, 954856, 954873, 1235145, 186844]

    for process_id in process_ids:
        xml_data, image_path = process_single_image(process_id, yolo_model, detection_labels, model_classification, device)
        if xml_data:
            save_xml(xml_data, f"{process_id}_processed.xml", "output2")

if __name__ == "__main__":
    main()
