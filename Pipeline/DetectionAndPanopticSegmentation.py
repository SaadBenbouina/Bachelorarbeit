import random
import cv2
import numpy as np
import requests
from ultralytics import YOLO
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch
from bs4 import BeautifulSoup
import io
from PIL import Image
import xml.etree.ElementTree as ET
from ShipLabelFilter import ShipLabelFilter
import os
import pycocotools.mask as mask_util
from detectron2.data import MetadataCatalog
import logging

# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_panoptic_model():
    """
    Einrichtung des panoptischen Segmentierungsmodells mit Detectron2.

    Rückgabe:
        tuple: Ein Tuple, das den Prädiktor und die Metadaten enthält.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)

    # Laden der Metadaten für COCO Panoptic Segmentation
    metadata = MetadataCatalog.get("coco_2017_val_panoptic_separated")

    return predictor, metadata

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
        # Berücksichtige nur Bounding-Boxen mit Confidence > 0.5 für die relevanten Labels
        if confidence > 0.5 and label in detection_labels:
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

def calculate_iou(mask, x1, y1, x2, y2):
    """
    Berechnet den Intersection over Union (IoU) zwischen einer Maske und einer Bounding-Box.

    Argumente:
        mask (np.ndarray): Die Segmentierungsmaske.
        x1 (int): Obere linke x-Koordinate der Bounding-Box.
        y1 (int): Obere linke y-Koordinate der Bounding-Box.
        x2 (int): Untere rechte x-Koordinate der Bounding-Box.
        y2 (int): Untere rechte y-Koordinate der Bounding-Box.

    Rückgabe:
        float: Der IoU-Wert.
    """
    # Sicherstellen, dass die Koordinaten innerhalb der Maskenabmessungen liegen
    height, width = mask.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    bbox_mask = np.zeros_like(mask, dtype=bool)
    bbox_mask[y1:y2, x1:x2] = True

    intersection = np.logical_and(mask, bbox_mask)
    union = np.logical_or(mask, bbox_mask)

    intersection_area = np.sum(intersection)
    union_area = np.sum(union)

    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

def select_best_mask_within_bbox(boat_scores, boat_boxes, panoptic_masks):
    """
    Wählt die beste Maske innerhalb jeder Bounding-Box basierend auf dem IoU aus.

    Argumente:
        boat_scores (np.ndarray): Scores der Boots-Masken.
        boat_boxes (list): Liste der Bounding-Box-Daten.
        panoptic_masks (np.ndarray): Array der panoptischen Masken.

    Rückgabe:
        list: Liste der ausgewählten Masken mit Scores und Labels.
    """
    selected_masks = []
    for i, box in enumerate(boat_boxes):
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        best_mask = None
        best_score = -1
        best_iou = 0

        # Finde die Maske mit dem höchsten IoU innerhalb der Bounding-Box
        for mask, score in zip(panoptic_masks, boat_scores):
            iou = calculate_iou(mask, x1, y1, x2, y2)

            # Wähle die Maske basierend auf IoU und Score
            if iou > best_iou:
                best_mask = mask
                best_score = score
                best_iou = iou

        if best_mask is not None:
            selected_masks.append((best_score, "boat", best_mask))

    return selected_masks

def apply_panoptic_segmentation(frame, panoptic_result, metadata, confidence_threshold=0.7, max_instances=0,
                                boxes_data=None):
    """
    Wendet panoptische Segmentierung auf den Frame an und zeichnet Masken.

    Argumente:
        frame (np.ndarray): Der Bild-Frame.
        panoptic_result (dict): Das Ergebnis der panoptischen Segmentierung.
        metadata (Metadata): Die Metadaten für die panoptische Segmentierung.
        confidence_threshold (float): Confidence-Schwelle für die Maskenauswahl.
        max_instances (int): Maximale Anzahl von Instanzen, die verarbeitet werden sollen.
        boxes_data (list, optional): Liste der Bounding-Box-Daten.

    Rückgabe:
        list: Liste der gezeichneten Masken mit Scores und Labels.
    """
    if boxes_data is None:
        boxes_data = []
    panoptic_seg, segments_info = panoptic_result["panoptic_seg"]  # Tuple aus Segmentierungsdaten und Segmentinformationen
    panoptic_seg = panoptic_seg.cpu().numpy()

    # Extrahiere und konvertiere Instanzdaten
    instances = panoptic_result["instances"]
    pred_classes = instances.pred_classes.cpu().numpy()
    pred_masks = instances.pred_masks.cpu().numpy()
    pred_scores = instances.scores.cpu().numpy()

    # Filtere gültige "boat"-Segmentierungen
    boat_class_id = metadata.thing_classes.index("boat")
    boat_mask_indices = np.where(
        (pred_scores > confidence_threshold) &
        (pred_classes == boat_class_id)
    )[0]

    # Extrahiere Boots-Masken und Scores
    boat_scores = pred_scores[boat_mask_indices]
    boat_masks = pred_masks[boat_mask_indices]

    # Wähle die besten Masken innerhalb der Bounding-Boxen aus
    drawn_masks = select_best_mask_within_bbox(
        boat_scores, boxes_data, boat_masks)

    # Begrenze die Anzahl der Instanzen basierend auf max_instances
    if max_instances > 0:
        drawn_masks = drawn_masks[:max_instances]

    # Zeichne die besten Masken auf das Bild
    for score, label, mask in drawn_masks:
        color = generate_unique_color()
        logger.info(f"Farbe für {label}: {color}")
        frame[mask] = (frame[mask] * 0.5 + color * 0.5).astype(np.uint8)

    return drawn_masks

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

def process_image(image, yolo_model, panoptic_model, metadata, detection_labels, url, photo_label, taking_time,
                  process_id, confidence_threshold=0.7, debug=True):
    """
    Verarbeitet ein einzelnes Bild für Detektion, Segmentierung und Klassifikation.

    Argumente:
        image (PIL.Image): Das zu verarbeitende Bild.
        yolo_model (YOLO): Das YOLO-Modell für die Objektdetektion.
        panoptic_model (DefaultPredictor): Das panoptische Segmentierungsmodell.
        metadata (Metadata): Metadaten für die panoptische Segmentierung.
        detection_labels (list): Zu detektierende Labels.
        url (str): URL des Bildes.
        photo_label (str): Label des Fotos.
        taking_time (str): Zeitpunkt der Aufnahme des Fotos.
        process_id (int): Prozess-ID zur Verfolgung.
        confidence_threshold (float): Confidence-Schwelle für Detektionen.
        debug (bool): Ob der Debug-Modus aktiviert werden soll.

    Rückgabe:
        tuple: XML-Element mit Metadaten und Pfad zum gespeicherten Bild.
    """
    image_np = np.array(image)
    width, height = image.size

    if debug:
        logger.info(f"Verarbeite Bild von URL: {url}")

    # Führe YOLO-Detektion durch
    yolo_result = yolo_model.predict(image_np)

    # Wende panoptische Segmentierung an
    panoptic_result = panoptic_model(image_np)

    # Zeichne YOLO-Detektionen und wende panoptische Segmentierung an
    detected_boxes, boxes_data = draw_yolo_detections(
        image_np, yolo_result, yolo_model, detection_labels)

    # Hole die gezeichneten Masken aus der panoptischen Segmentierung
    drawn_masks = apply_panoptic_segmentation(image_np, panoptic_result, metadata,
                                              confidence_threshold=confidence_threshold,
                                              max_instances=detected_boxes, boxes_data=boxes_data)

    # Speichere das verarbeitete Bild
    image_path = save_image(image_np, "/private/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/Pipeline/output",
                            f"{process_id}_processed.jpg")

    # Erstelle XML-Struktur zum Speichern der Bildmetadaten
    image_metadata = ET.Element("image", id=str(process_id), category=f"{photo_label}", date=f"{taking_time}",
                                width=str(width), height=str(height))

    # Speichere die Bounding-Boxen im XML
    for box in boxes_data:
        box_element = ET.SubElement(
            image_metadata, "Bounding_box", label=box['label'], confidence=f"{box['confidence']:.2f}")
        box_element.set("x1", str(box['x1']))
        box_element.set("y1", str(box['y1']))
        box_element.set("x2", str(box['x2']))
        box_element.set("y2", str(box['y2']))
        logger.info(
            f"Bounding-Box hinzugefügt für {box['label']} mit Confidence {box['confidence']:.2f}")

    # Speichere die gezeichneten Masken im XML
    for score, label, mask in drawn_masks:
        rle = mask_to_rle(mask)  # Konvertiere die Maske in RLE
        mask_element = ET.SubElement(
            image_metadata, "mask", label=label, source="auto")
        rle_str = ', '.join(str(count) for count in rle["counts"])
        mask_element.set("rle", rle_str)
        logger.info(f"Maske hinzugefügt für {label} mit Confidence {score:.2f}")

    # Freigeben ungenutzter Variablen
    del yolo_result, panoptic_result, detected_boxes, boxes_data, drawn_masks

    # Bild in einem Fenster mit OpenCV anzeigen (optional)
    if debug:
        cv2.imshow(f"Verarbeitetes Bild: {photo_label}", image_np)
        cv2.waitKey(0)  # Warte auf Tastendruck zum Schließen des Fensters
        cv2.destroyAllWindows()  # Schließe das Fenster

    return image_metadata, image_path

def scrape_and_process_ship_images(process_id, yolo_model, panoptic_model, metadata, detection_labels):
    """
    Scrapt Bilder von ShipSpotting.com und verarbeitet sie.

    Argumente:
        process_id (int): Die ID des zu verarbeitenden Bildes.
        yolo_model (YOLO): Das YOLO-Modell für die Objektdetektion.
        panoptic_model (DefaultPredictor): Das panoptische Segmentierungsmodell.
        metadata (Metadata): Metadaten für die panoptische Segmentierung.
        detection_labels (list): Zu detektierende Labels.

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
        label_divs = soup.find_all('div',
                                   class_='InformationItem__InfoItemStyle-sc-r4h3tv-0 hfSVPp information-item '
                                          'summary-photo__card-general__label')
        label_text = ""
        taken_time = ""
        for div in label_divs:
            information_title = div.find(
                'span', class_='information-item__title')
            if information_title and information_title.text.strip() == "Captured:":
                taken_time_value = div.find(
                    'span', class_='information-item__value')
                if taken_time_value:
                    taken_time = taken_time_value.text.strip()
            if information_title and information_title.text.strip() == "Photo Category:":
                label_value = div.find(
                    'span', class_='information-item__value')
                if label_value:
                    # Extrahiere den Kategorie-Text
                    label_text = label_value.text.strip()
                    break

        if not image_urls:
            logger.warning(f"Kein Bild gefunden für process_id: {process_id}")
            return None, None

        image_url = image_urls[0]
        if image_url.startswith('/'):
            image_url = 'https://www.shipspotting.com' + image_url

        logger.info(f"Lade Bild herunter von: {image_url}")
        image_response = session.get(image_url, headers=headers)
        image = Image.open(io.BytesIO(image_response.content))

        label = ShipLabelFilter.filter_label(label_text)
        logger.info(f"Verarbeite Schiffsbild mit Label: {label[0]}")

        xml_data, image_path = process_image(image, yolo_model, panoptic_model, metadata, detection_labels, image_url,
                                             label[0], taken_time, process_id, debug=True)

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
    logger.info(f"XML-Daten gespeichert unter {file_path}")
    return file_path

def main():
    """
    Hauptfunktion zur Ausführung des Scraping- und Verarbeitungspipelines.
    """
    # Lade das YOLO-Modell und verschiebe es auf das entsprechende Gerät
    yolo_model = YOLO("../YoloModel/boat_detection_yolo_model_new3/weights/best.pt")
    yolo_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    panoptic_model, metadata = setup_panoptic_model()
    detection_labels = ["boat"]

    process_id = 1335020  # Beispielhafte ShipSpotting-Bild-ID
    xml_data, image_path = scrape_and_process_ship_images(
        process_id, yolo_model, panoptic_model, metadata, detection_labels)

    if xml_data:
        save_xml(xml_data, f"{process_id}_processed.xml",
                 "/private/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/Pipeline/output")


if __name__ == "__main__":
    main()
