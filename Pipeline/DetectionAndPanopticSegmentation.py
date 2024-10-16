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
    Wendet die SAM-Segmentierung auf die Bounding-Boxen an, indem positive und negative Punkte verwendet werden.

    Argumente:
        frame (np.ndarray): Das Originalbild.
        predictor (SamPredictor): Das SAM-Predictor-Objekt.
        boxes_data (list): Liste der Bounding-Box-Daten.

    Rückgabe:
        list: Liste der gezeichneten Masken mit Scores und Labels.
    """
    drawn_masks = []
    predictor.set_image(frame)
    image_height, image_width = frame.shape[:2]

    for box in boxes_data:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        input_box = np.array([x1, y1, x2, y2])

        # Positive Punkte
        positive_points = np.array([
            [x1 + (x2 - x1) * 0.5, y1 + (y2 - y1) * 0.6],    # Mittlerer Punkt
            [x1 + (x2 - x1) * 0.5, y1 + (y2 - y1) * 0.85],   # Unterer mittlerer Bereich
            [x1 + (x2 - x1) * 0.7, y1 + (y2 - y1) * 0.85],   # Unterer rechter Bereich
        ])

        # Negative Punkte
        neg_x = x1 + (x2 - x1) * 0.5
        neg_y = y1 - (y2 - y1) * 0.1
        neg_y2 = y1 + (y2 - y1) * 0.05
        neg_x2 = x1 + (x2 - x1) * 0.05
        neg_x3 =x1 + (x2 - x1) * 0.35
        neg_y3 =  y1 + (y2 - y1) * 0.2
        neg_x4 =x1 + (x2 - x1) * 0.75
        neg_y4 = y1 + (y2 - y1) * 0.2
        neg_x5 =x1 + (x2 - x1) * 0.9
        neg_y5 =  y1 + (y2 - y1) * 0.15
        neg_x6 =x1 + (x2 - x1) * 0.1
        neg_y6 =  y1 + (y2 - y1) * 0.15

        # Prüfen und Anpassen der negativen Punkte, um sicherzustellen, dass sie innerhalb des Bildes liegen
        if neg_y < 0:
            neg_y = y2 + (y2 - y1) * 0.1
            if neg_y > image_height - 1:
                neg_y = image_height - 1

        if neg_x2 < 0:
            neg_x2 = 0

        negative_points = np.array([
            [neg_x, neg_y],
            [neg_x2, neg_y2],
            [neg_x3, neg_y3],
            [neg_x4, neg_y4],
            [neg_x5, neg_y5],
            [neg_x6, neg_y6]

        ])

        # Kombiniere die Punkte
        point_coords = np.vstack([positive_points, negative_points])
        point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))

        # Generiere die Maske mit SAM
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
    for score, label, mask in drawn_masks:
        rle = mask_to_rle(mask)  # Konvertiere die Maske in RLE
        mask_element = ET.SubElement(
            image_metadata, "mask", label=label, source="SAM")
        rle_str = ', '.join(str(count) for count in rle["counts"])
        mask_element.set("rle", rle_str)
    """"
        cv2.imshow(f"Verarbeitetes Bild: {photo_label}", image_np)
        cv2.waitKey(0)  # Warte auf Tastendruck zum Schließen des Fensters
        cv2.destroyAllWindows()  # Schließe das Fenster
    """
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

def save_xml(xml_data, file_name="output1.xml", path=""):
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

def process_single_image(process_id, yolo_model, sam_predictor, detection_labels, model_classification, device):
    """
    Wrapper function to process a single image by its process_id. Used in multiprocessing.

    Args:
        process_id (int): The process ID of the image to scrape and process.
        yolo_model (YOLO): The YOLO object detection model.
        sam_predictor (SamPredictor): The SAM predictor model.
        detection_labels (list): List of detection labels for YOLO.
        model_classification (nn.Module): The classification model.
        device (torch.device): The device (CPU/GPU).

    Returns:
        tuple: XML data and path to the processed image.
    """
    return scrape_and_process_ship_images(process_id, yolo_model, sam_predictor, detection_labels, model_classification, device)

def main():
    """
    Hauptfunktion zur Ausführung des Scraping- und Verarbeitungspipelines.
    """
    # Definieren Sie das Gerät
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lade das YOLO-Modell und verschiebe es auf das entsprechende Gerät
    yolo_model = YOLO("/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/boat_detection_yolo_model_new6/weights/best.pt")
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

    # Initialisiere das SAM-Modell
    sam_predictor = setup_sam_model()
    detection_labels = ["boat"]

    # Define the list of process IDs (e.g., a list of ship spotting image IDs)
    process_ids = [46844]  # Example of multiple IDs

    # Use a multiprocessing pool to process the images in parallel
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Create a partial function that passes the static arguments to the multiprocessing worker
    args = [(process_id, yolo_model, sam_predictor, detection_labels, model_classification, device) for process_id in process_ids]

    # Run the processing in parallel
    results = pool.starmap(process_single_image, args)

    # Save results
    for xml_data, image_path in results:
        if xml_data:
            process_id = xml_data.attrib['id']
            save_xml(xml_data, f"{process_id}_processed.xml", "output1")

    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
