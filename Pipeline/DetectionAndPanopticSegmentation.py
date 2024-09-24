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


# Importieren der SAM-Module
from segment_anything import sam_model_registry, SamPredictor

from Pipeline.Mapper import map_number_to_ship

# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_sam_model():
    """
    Einrichtung des SAM-Modells.

    Rückgabe:
        SamPredictor: Ein SAM-Predictor-Objekt.
    """
    # Pfad zum vortrainierten SAM-Modell (stellen Sie sicher, dass dieser korrekt ist)
    sam_checkpoint = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/sam_checkpoint/sam_vit_h_4b8939.pth"  # Verwenden Sie den Pfad zu Ihrem heruntergeladenen SAM-Modell
    model_type = "vit_h"

    # Laden des SAM-Modells
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = SamPredictor(sam)

    return predictor

def draw_yolo_detections(frame, yolo_result, yolo_model, detection_labels):
    """
    Zeichnet YOLO-Detektionen auf den Frame und sammelt Bounding-Box-Daten.

    Argumente:
        frame (np.ndarray): Der Bild-Frame.
        yolo_result (list): Die YOLO-Detektionsergebnisse.
        yolo_model (YOLO): Das YOLO-Modell, das für die Detektion verwendet wird.
        detection_labels (list): Labels, die detektiert werden sollen.

    Rückgabe:
        tuple: Anzahl der detektierten Boxen und Liste der Box-Daten.
    """
    detected_boxes = 0
    detections = yolo_result[0]
    boxes_data = []

    # Filtere Detektionen für die angegebenen Labels
    for i, box in enumerate(detections.boxes):
        class_id = int(box.cls[0])
        label = yolo_model.names[class_id]
        confidence = box.conf[0]
        # Berücksichtige nur Bounding-Boxen mit Confidence > 0.6 für die relevanten Labels
        if confidence > 0.6 and label in detection_labels:
            detected_boxes += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            label_text = f'{label} {confidence:.2f}'
            # Zeichne die Bounding-Box und das Label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Füge Label und Confidence hinzu
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            logger.info(
                f"Ausgewählt: {label} mit Confidence {confidence:.2f} bei Koordinaten: ({x1}, {y1}), ({x2}, {y2})")
            boxes_data.append({
                'label': label,
                'confidence': confidence,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            })

    return detected_boxes, boxes_data

def generate_unique_color():
    """
    Generiert eine eindeutige Farbe für jedes Objekt.

    Rückgabe:
        np.ndarray: Ein Array, das eine Farbe im RGB-Format repräsentiert.
    """
    return np.array([random.randint(0, 255), random.randint(
        0, 255), random.randint(0, 255)], dtype=np.uint8)

def mask_to_rle(mask):
    """
    Konvertiert eine Maske in das RLE-Format.

    Argumente:
        mask (np.ndarray): Die zu konvertierende Maske.

    Rückgabe:
        str: Der RLE-String.
    """
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle

def save_image(image_np, path, filename):
    """
    Speichert das verarbeitete Bild im angegebenen Pfad.

    Argumente:
        image_np (np.ndarray): Das Bild-Array.
        path (str): Der Verzeichnispfad zum Speichern des Bildes.
        filename (str): Der Dateiname für das Bild.

    Rückgabe:
        str: Der vollständige Pfad zum gespeicherten Bild.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = os.path.join(path, filename)
    cv2.imwrite(image_path, image_np)  # Speichere das Bild
    return image_path

def apply_sam_segmentation(frame, predictor, boxes_data):
    """
    Applies SAM segmentation to the bounding boxes using point prompts.

    Args:
        frame (np.ndarray): The original image.
        predictor (SamPredictor): The SAM predictor object.
        boxes_data (list): List of bounding box data.

    Returns:
        list: List of drawn masks with scores and labels.
    """
    drawn_masks = []

    # Bereite das Bild für SAM vor
    predictor.set_image(frame)

    for box in boxes_data:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

        # SAM erwartet die Eingabe in der Form [x0, y0, x1, y1]
        input_box = np.array([x1, y1, x2, y2])

        # Generate point prompts
        # Positive point at the center of the lower half of the bounding box
        pos_x = x1 + (x2 - x1) / 2
        pos_y = y1 + 3 * (y2 - y1) / 4  # Lower half
        positive_point = np.array([[pos_x, pos_y]])

        # Negative point at the center of the top edge of the bounding box
        neg_x = x1 + (x2 - x1) / 2
        neg_y = y1 + (y2 - y1) / 4  # Upper quarter
        negative_point = np.array([[neg_x, neg_y]])

        # Combine points
        point_coords = np.vstack([positive_point, negative_point])
        point_labels = np.array([1, 0])  # 1 for positive, 0 for negative

        # Generate the mask with SAM
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box[None, :],
            multimask_output=False
        )

        mask = masks[0]
        score = scores[0]

        # Zeichnen Sie die Maske auf das Bild
        color = generate_unique_color()
        mask_bool = mask.astype(bool)
        frame[mask_bool] = (frame[mask_bool] * 0.5 + color * 0.5).astype(np.uint8)

        drawn_masks.append((score, box['label'], mask_bool))

        logger.info(f"Maske hinzugefügt für {box['label']} mit SAM-Score {score:.2f}")

    return drawn_masks

def process_image(image, yolo_model, sam_predictor, detection_labels, photo_label,
                  process_id):
    """
    Verarbeitet ein einzelnes Bild für Detektion, Segmentierung und Klassifikation.

    Argumente:
        image (PIL.Image): Das zu verarbeitende Bild.
        yolo_model (YOLO): Das YOLO-Modell für die Objektdetektion.
        sam_predictor (SamPredictor): Der SAM-Predictor.
        detection_labels (list): Zu detektierende Labels.
        url (str): URL des Bildes.
        photo_label (str): Label des Fotos.
        taking_time (str): Zeitpunkt der Aufnahme des Fotos.
        process_id (int): Prozess-ID zur Verfolgung.
        confidence_threshold (float): Confidence-Schwelle für Detektionen.

    Rückgabe:
        tuple: XML-Element mit Metadaten und Pfad zum gespeicherten Bild.
    """
    image_np = np.array(image)
    width, height = image.size

    # Führe YOLO-Detektion durch
    yolo_result = yolo_model.predict(image_np)

    # Zeichne YOLO-Detektionen und sammle die Bounding-Box-Daten
    detected_boxes, boxes_data = draw_yolo_detections(
        image_np, yolo_result, yolo_model, detection_labels)

    if detected_boxes == 0:
        logger.warning("Keine Objekte in YOLO-Detektion gefunden.")
        return None, None

    # Wende SAM-Segmentierung auf die Bounding-Boxen an
    drawn_masks = apply_sam_segmentation(
        image_np, sam_predictor, boxes_data)

    # Speichere das verarbeitete Bild
    image_path = save_image(image_np, "output",
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
    for score, label, mask in drawn_masks:
        rle = mask_to_rle(mask)  # Konvertiere die Maske in RLE
        mask_element = ET.SubElement(
            image_metadata, "mask", label=label, source="SAM")
        rle_str = ', '.join(str(count) for count in rle["counts"])
        mask_element.set("rle", rle_str)

        cv2.imshow(f"Verarbeitetes Bild: {photo_label}", image_np)
        cv2.waitKey(0)  # Warte auf Tastendruck zum Schließen des Fensters
        cv2.destroyAllWindows()  # Schließe das Fenster

    return image_metadata, image_path

def classify_image(image, model_classification, device):
    """
    Klassifiziert ein Bild mit dem gegebenen Klassifikationsmodell.

    Argumente:
        image (PIL.Image): Das zu klassifizierende Bild.
        model_classification (nn.Module): Das Klassifikationsmodell.
        device (torch.device): Das verwendete Gerät (CPU oder GPU).

    Rückgabe:
        int: Die vorhergesagte Klassen-ID.
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


def scrape_and_process_ship_images(process_id, yolo_model, sam_predictor, detection_labels, model_classification, device):
    """
    Scrapt Bilder von ShipSpotting.com und verarbeitet sie.

    Argumente:
        process_id (int): Die ID des zu verarbeitenden Bildes.
        yolo_model (YOLO): Das YOLO-Modell für die Objektdetektion.
        sam_predictor (SamPredictor): Der SAM-Predictor.
        detection_labels (list): Zu detektierende Labels.
        model_classification (nn.Module): Das Klassifikationsmodell.
        device (torch.device): Das verwendete Gerät (CPU oder GPU).

    Rückgabe:
        tuple: XML-Daten und Pfad zum verarbeiteten Bild.
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
        """"
        label_divs = soup.find_all('div',
                                   class_='InformationItem__InfoItemStyle-sc-r4h3tv-0 hfSVPp information-item '
                                         'summary-photo__card-general__label')
        label_text = ""
        for div in label_divs:
            information_title = div.find(
                'span', class_='information-item__title')
            if information_title and information_title.text.strip() == "Photo Category:":
                label_value = div.find(
                    'span', class_='information-item__value')
                if label_value:
                    # Extrahiere den Kategorie-Text
                    label_text = label_value.text.strip()
                    break
                    
        label = ShipLabelFilter.filter_label(label_text)
        logger.info(f"Verarbeite Schiffsbild mit Label: {label[0]}")
        """
        if not image_urls:
            logger.warning(f"Kein Bild gefunden für process_id: {process_id}")
            return None, None

        image_url = image_urls[0]
        if image_url.startswith('/'):
            image_url = 'https://www.shipspotting.com' + image_url

        image_response = session.get(image_url, headers=headers)
        image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
        label_pred = map_number_to_ship(classify_image(image, model_classification, device))
        logger.info(f"Schiffkategory ist : {label_pred}")

        xml_data, image_path = process_image(image, yolo_model, sam_predictor, detection_labels,
                                             label_pred, process_id)

        # Freigeben ungenutzter Variablen
        del response, soup, image_response, image

        return xml_data, image_path
    except Exception as e:
        logger.error(f"Fehler beim Verarbeiten des Bildes: {e}")
        return None, None
    finally:
        session.close()

def save_xml(xml_data, file_name="output.xml", path=""):
    """
    Speichert XML-Daten mit Detektions-, Segmentierungs- und Klassifikationsdaten.

    Argumente:
        xml_data (ET.Element): Die zu speichernden XML-Daten.
        file_name (str): Der Dateiname für die XML-Datei.
        path (str): Der Verzeichnispfad zum Speichern der XML-Datei.

    Rückgabe:
        str: Der vollständige Pfad zur gespeicherten XML-Datei.
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

def main():
    """
    Hauptfunktion zur Ausführung des Scraping- und Verarbeitungspipelines.
    """
    # Definieren Sie das Gerät
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lade das YOLO-Modell und verschiebe es auf das entsprechende Gerät
    yolo_model = YOLO("../YoloModel/boat_detection_yolo_model_new3/weights/best.pt")
    yolo_model.to(device)

    # Instanziiere das Klassifikationsmodell
    num_classes = 16
    model_classification = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model_classification.fc.in_features
    model_classification.fc = nn.Linear(num_ftrs, num_classes)

    # Pfad zu den gespeicherten Gewichten
    model_save_path = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/ship_classification_resnet50.pth"

    # Laden der state_dict
    state_dict = torch.load(model_save_path, map_location=device)
    # Laden der state_dict in das Modell
    model_classification.load_state_dict(state_dict)
    model_classification.to(device)
    model_classification.eval()  # Setzt das Modell in den Evaluationsmodus

    # Initialisiere das SAM-Modell
    sam_predictor = setup_sam_model()
    detection_labels = ["boat"]

    process_id = 954847  # Beispielhafte ShipSpotting-Bild-ID
    xml_data, image_path = scrape_and_process_ship_images(
        process_id, yolo_model, sam_predictor, detection_labels, model_classification, device)

    if xml_data:
        save_xml(xml_data, f"{process_id}_processed.xml",
                 "output")


if __name__ == "__main__":
    main()
