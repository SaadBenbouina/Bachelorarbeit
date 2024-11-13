import numpy as np
import xml.etree.ElementTree as ET
import pycocotools.mask as mask_util
import random
import cv2
import requests
import torchvision.transforms as transforms
from ultralytics import YOLO
import torch
from torchvision import models
import torch.nn as nn
from bs4 import BeautifulSoup
import io
from PIL import Image
import os
import logging
import multiprocessing

from Pipeline.Mapper import map_number_to_ship

# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def draw_yolo_detections(frame, yolo_result, yolo_model, detection_labels):
    """
    Zeichnet YOLO-Detektionen auf den Frame und sammelt Bounding-Box-Daten.
    """
    detected_boxes = 0
    detections = yolo_result[0]
    boxes_data = []

    # Filtere Detektionen für die angegebenen Labels
    for i, box in enumerate(detections.boxes):
        class_id = int(box.cls[0])
        label = yolo_model.names[class_id]
        confidence = box.conf[0]
        # Berücksichtige nur Bounding-Boxen mit Confidence > 0.5 für die relevanten Labels
        if confidence > 0.4 and label in detection_labels:
            detected_boxes += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            label_text = f'{label} {confidence:.2f}'
            # Zeichne die Bounding-Box und das Label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Füge Label und Confidence hinzu
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            boxes_data.append({
                'label': label,
                'confidence': confidence,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            })

    return detected_boxes, boxes_data

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

def apply_yolo_segmentation(frame, yolo_model, boxes_data):
    """
    Wendet die YOLOv8-Segmentierung auf die Bounding-Boxen an.
    """
    drawn_masks = []
    image_height, image_width = frame.shape[:2]

    for box in boxes_data:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

        # Sicherstellen, dass die Koordinaten innerhalb der Bildgrenzen liegen
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)

        cropped_frame = frame[y1:y2, x1:x2]
        cropped_height, cropped_width = cropped_frame.shape[:2]

        if cropped_height == 0 or cropped_width == 0:
            logger.warning("Ungültige Dimensionen des ausgeschnittenen Bereichs.")
            continue

        # Führen Sie die Segmentierung mit YOLOv8 auf dem ausgeschnittenen Bereich durch
        yolo_result = yolo_model.predict(cropped_frame, task='segment')

        # Überprüfen Sie, ob eine Maske erkannt wurde
        detections = yolo_result[0]
        if detections.masks is None or len(detections.masks.data) == 0:
            continue  # Keine Maske erkannt

        # Nehmen Sie die erste erkannte Maske
        mask = detections.masks.data[0].cpu().numpy()

        # Skalieren Sie die Maske auf die Größe des ausgeschnittenen Bereichs
        mask_resized = cv2.resize(mask, (cropped_width, cropped_height), interpolation=cv2.INTER_NEAREST)
        mask_resized = mask_resized.astype(bool)

        # Platzieren Sie die skalierte Maske in der Gesamtmaske
        full_mask = np.zeros((image_height, image_width), dtype=bool)
        full_mask[y1:y2, x1:x2] = mask_resized

        # Zeichnen Sie die Maske auf das Originalbild
        color = generate_unique_color()
        frame[full_mask] = (frame[full_mask] * 0.5 + color * 0.5).astype(np.uint8)

        # Fügen Sie die Maske zu den gezeichneten Masken hinzu
        drawn_masks.append((box['confidence'], box['label'], full_mask))

        logger.info(f"Maske hinzugefügt für {box['label']} mit YOLO-Segmentierung")

    return drawn_masks

def process_image(image, yolo_model, detection_labels, photo_label, process_id):
    """
    Verarbeitet ein einzelnes Bild für Detektion, Segmentierung und Klassifikation.
    """
    image_np = np.array(image)
    width, height = image.size

    # Führe YOLO-Detektion und Segmentierung durch
    yolo_result = yolo_model.predict(image_np, task='segment')

    # Zeichne YOLO-Detektionen und sammle die Bounding-Box-Daten
    detected_boxes, boxes_data = draw_yolo_detections(
        image_np, yolo_result, yolo_model, detection_labels)

    if detected_boxes == 0:
        logger.warning("Keine Objekte in YOLO-Detektion gefunden.")
        return None, None

    # Wende YOLO-Segmentierung auf die Bounding-Boxen an
    drawn_masks = apply_yolo_segmentation(
        image_np, yolo_model, boxes_data)

    # Speichere das verarbeitete Bild
    image_path = save_image(image_np, "output1",
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

def scrape_and_process_ship_images(process_id, yolo_model, detection_labels, model_classification, device):
    """
    Scrapt Bilder von ShipSpotting.com und verarbeitet sie.
    """
    url_prefix = 'https://www.shipspotting.com/photos/'
    url = f"{url_prefix}{str(process_id).zfill(7)}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                      ' Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.shipspotting.com',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    # Verwende eine Session, um Verbindungen zu persistieren
    session = requests.Session()
    try:
        response = session.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Logik zum Scrapen der Bilder
        divs = soup.find_all('div', class_='summary-photo__image-row__image')
        image_urls = [div.find('img')['src']
                      for div in divs if div.find('img')]

        if not image_urls:
            logger.warning(f"Kein Bild gefunden für process_id: {process_id}")
            return None, None

        image_url = image_urls[0]
        if image_url.startswith('/'):
            image_url = 'https://www.shipspotting.com' + image_url

        image_response = session.get(image_url, headers=headers)
        image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
        label_pred = map_number_to_ship(classify_image(image, model_classification, device))
        logger.info(f"Schiffkategorie ist : {label_pred}")

        xml_data, image_path = process_image(image, yolo_model, detection_labels,
                                             label_pred, process_id)

        # Freigeben ungenutzter Variablen
        del response, soup, image_response, image

        return xml_data, image_path
    except Exception as e:
        logger.error(f"Fehler beim Verarbeiten des Bildes: {e}")
        return None, None
    finally:
        session.close()

def save_xml(xml_data, file_name="output1.xml", path=""):
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

def process_single_image(process_id, yolo_model, detection_labels, model_classification, device):
    """
    Wrapper-Funktion zum Verarbeiten eines einzelnen Bildes anhand seiner process_id. Wird in Multiprocessing verwendet.
    """
    return scrape_and_process_ship_images(process_id, yolo_model, detection_labels, model_classification, device)

def main():
    """
    Hauptfunktion zur Ausführung des Scraping- und Verarbeitungspipelines.
    """
    # Definieren Sie das Gerät
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lade das YOLOv8-Segmentierungsmodell und verschiebe es auf das entsprechende Gerät
    yolo_model = YOLO("/private/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/boat_segmentation/weights/best.pt")  # Verwenden Sie Ihr trainiertes Modell oder ein vortrainiertes Modell
    yolo_model.to(device)

    # Instanziiere das Klassifikationsmodell
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

    # Definieren Sie die Liste der process_ids (z.B. eine Liste von ShipSpotting-Bild-IDs)
    process_ids = [76844, 96844, 126744, 154644, 154859, 933373, 954856, 954873, 1235145, 186844]  # Beispiel für mehrere IDs

    # Verwenden Sie einen Multiprocessing-Pool, um die Bilder parallel zu verarbeiten
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Erstellen Sie eine Argumentliste für die Multiprocessing-Worker
    args = [(process_id, yolo_model, detection_labels, model_classification, device) for process_id in process_ids]

    # Führen Sie die Verarbeitung parallel aus
    results = pool.starmap(process_single_image, args)

    # Ergebnisse speichern
    for xml_data, image_path in results:
        if xml_data:
            process_id = xml_data.attrib['id']
            save_xml(xml_data, f"{process_id}_processed.xml", "output1")

    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
