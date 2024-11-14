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

def process_image(image, yolo_model, detection_labels, model_classification, device, process_id):
    """
    Verarbeitet ein einzelnes Bild für Detektion, Segmentierung und Klassifikation für alle erkannten Boote.
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
            cropped_image = image.crop((x1, y1, x2, y2))  # Erstelle ein zugeschnittenes Bild des Bootes
            label_pred = classify_and_log(cropped_image, model_classification, device)  # Klassifikation für das Boot

            label_text = f'{label_pred} {confidence:.2f}'
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            boxes_data.append({
                'label': label_pred,
                'confidence': confidence,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            })

            # Wenn eine Maske vorhanden ist, zeichnen Sie sie
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
    image_metadata = ET.Element("image", id=str(process_id), width=str(width), height=str(height))

    # Speichere Bounding-Boxen und Masken im XML
    for box in boxes_data:
        box_element = ET.SubElement(image_metadata, "Bounding_box", label=box['label'], confidence=f"{box['confidence']:.2f}")
        box_element.set("x1", str(box['x1']))
        box_element.set("y1", str(box['y1']))
        box_element.set("x2", str(box['x2']))
        box_element.set("y2", str(box['y2']))

    for confidence, label, mask in drawn_masks:
        rle = mask_to_rle(mask)
        mask_element = ET.SubElement(image_metadata, "mask", label=label, source="YOLOv8")
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

    # Segmentierung und Klassifikation aller Boote im Bild
    xml_data, image_path = process_image(image, yolo_model, detection_labels, model_classification, device, process_id)
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
    process_ids = [
        88064, 36865, 24590, 59410, 26651, 36896, 81958, 59432, 12338, 81972,
        59451, 61503, 90182, 45127, 53330, 65619, 14419, 49240, 57463, 88183,
        69761, 30858, 51338, 59538, 22675, 49299, 88225, 32932, 14500, 49317,
        34983, 57526, 53432, 20668, 84156, 35009, 69827, 75993, 14558, 92385,
        22775, 47354, 94460, 10502, 28950, 78107, 94492, 12577, 53538, 41253,
        20774, 14632, 39217, 78129, 74036, 88380, 98643, 98656, 69985, 43372,
        84339, 39284, 86391, 84348, 14733, 96662, 20887, 84387, 14769, 31156,
        25024, 31171, 16848, 35286, 72151, 72152, 92638, 43487, 16864, 82406,
        78312, 72169, 92648, 70124, 74222, 74224, 86515, 92663, 53756, 94717,
        84478, 80397, 33298, 55834, 41501, 72225, 23078, 33320, 55849, 86569,
        86581, 59959, 62007, 29241, 98877, 82505, 62027, 53845, 92768, 72290,
        62052, 35432, 64114, 64117, 47738, 84606, 70272, 49794, 43650, 94851,
        51845, 62099, 43669, 31386, 31388, 96928, 19106, 39586, 64164, 78500,
        41638, 62127, 17073, 10934, 58040, 23238, 86734, 62159, 58070, 37601,
        56035, 74468, 76518, 23273, 35565, 15086, 99058, 51956, 92925, 21245,
        80643, 49923, 76547, 54024, 58124, 43790, 92949, 62246, 72488, 33580,
        35629, 13129, 41823, 68451, 11115, 50032, 99188, 58229, 13180, 45950,
        23426, 64386, 64391, 72586, 29578, 62346, 66448, 25498, 23454, 58275,
        17323, 27565, 37825, 46017, 15315, 11220, 72662, 29660, 62432, 82913,
        29667, 93163, 82935, 15355, 99324, 70653, 44031, 15371, 97293, 76821,
        27670, 64535, 50201, 19483, 50205, 64554, 50242, 64583, 66650, 27739,
        81001, 70762, 74859, 31852, 64623, 27764, 33919, 68738, 52357, 93318,
        46224, 87190, 23703, 48282, 72859, 25762, 11442, 58558, 95423, 87233,
        72900, 81095, 81098, 95439, 31953, 54494, 21729, 81121, 46319, 44283,
        25863, 46348, 44301, 42269, 27934, 89375, 72995, 97574, 75049, 83241,
        32047, 93490, 75059, 77137, 70996, 68949, 50526, 17758, 77152, 32127,
        42367, 97665, 42371, 66950, 30087, 58760, 52620, 69005, 13724, 13727,
        50613, 71098, 81339, 26047, 67008, 50626, 15813, 85448, 30154, 62923,
        50638, 67032, 64985, 34269, 38367, 30179, 91623, 85480, 83437, 38383,
        69104, 97784, 69116, 56841, 22028, 99853, 65052, 87583, 22049, 34342,
        97841, 24115, 44610, 58948, 63054, 46671, 95831, 91742, 93798, 40557,
        71281, 93828, 22153, 36490, 79502, 50836, 50839, 77468, 50844, 28320,
        61105, 40625, 81587, 63159, 32453, 69319, 48845, 81616, 18130, 65239,
        52960, 59107, 46820, 12007, 83692, 12012, 93933, 79603, 71413, 28412,
        93951, 87821, 38675, 53017, 89886, 63264, 32544, 42787, 28454, 22320,
        71472, 30513, 96052, 79670, 69433, 55100, 89920, 44865, 28485, 94021,
        75594, 46927, 57168, 94032, 67433, 85869, 34678, 20351, 79750, 77703,
        65422, 40849, 51092, 96149, 69526, 71574, 73628, 61352, 75694, 40879,
        14259, 90047, 44993, 69569, 61382, 16329, 26578, 73685, 47064, 65500,
        26594, 18405, 83947, 69617, 98290, 28659, 71668, 43002, 43006, 49151
    ]
    for process_id in process_ids:
        xml_data, image_path = process_single_image(process_id, yolo_model, detection_labels, model_classification, device)
        if xml_data:
            save_xml(xml_data, f"{process_id}_processed.xml", "output2")

if __name__ == "__main__":
    main()
