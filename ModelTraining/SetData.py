import os

import numpy as np
import cv2
import logging

from ultralytics import YOLO
from Pipeline.ship_image_scraper import scrape_image_selenium as scrape_image  # Importieren Sie Ihre Bildabruffunktion

# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_image(image_np, path, filename):
    """
    Speichert das Bild im angegebenen Pfad.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = os.path.join(path, filename)
    success = cv2.imwrite(image_path, image_np)
    if not success:
        logger.error(f"Fehler beim Speichern des Bildes unter: {image_path}")
        return None
    return image_path

def save_label_txt(label_data, path, filename):
    """
    Speichert die Annotationsdaten in einer .txt-Datei im angegebenen Pfad.
    Format für YOLOv8-Segmentierung: class x1 y1 x2 y2 ... xn yn
    """
    if not os.path.exists(path):
        os.makedirs(path)
    label_path = os.path.join(path, filename)
    with open(label_path, 'w') as f:
        for line in label_data:
            f.write(f"{line}\n")
    return label_path

def create_label_txt(annotations):
    """
    Erstellt die Inhalte für die .txt Annotationsdatei für YOLOv8-Segmentierung.
    """
    label_data = []
    for ann in annotations:
        category_id = ann["category_id"]
        segmentation = ann.get("segmentation", [])
        if segmentation:
            segmentation_str = ' '.join([f"{coord:.6f}" for coord in segmentation])
            label_line = f"{category_id} {segmentation_str}"
            label_data.append(label_line)
    return label_data

def process_image(image, yolo_model, detection_labels, process_id, class_mapping, images_dir, conf_threshold=0.2):
    """
    Verarbeitet ein einzelnes Bild: Detektion, Segmentierung und Speichern der Ergebnisse.
    """
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    # Speichere das rohe Bild
    raw_image_filename = f"{process_id}.jpg"
    raw_image_path = save_image(image_np, images_dir, raw_image_filename)
    if raw_image_path is None:
        return None, None, None, None

    # Führe YOLO-Detektion und Segmentierung durch
    yolo_result = yolo_model.predict(image_np, task='segment', conf=conf_threshold)
    detected_boxes = 0
    yolo_annotations = []

    detections = yolo_result[0]

    for i, box in enumerate(detections.boxes):
        class_id = int(box.cls[0])
        label = yolo_model.names[class_id]
        confidence = box.conf[0]

        if confidence > conf_threshold and label in detection_labels:
            detected_boxes += 1

            # Verarbeite die Segmentierungsmaske
            if detections.masks is not None:
                mask = detections.masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_resized = mask_resized.astype(np.uint8)

                # Extrahiere Konturen aus der Maske
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    # Konturpunkte flachmachen und normalisieren
                    segmentation = largest_contour.reshape(-1).astype(float)
                    segmentation = [coord / width if idx % 2 == 0 else coord / height for idx, coord in enumerate(segmentation)]
                else:
                    segmentation = []

                # Füge Annotation hinzu
                annotation = {
                    "category_id": class_mapping.get(label, 0),
                    "segmentation": segmentation,
                    "width": width,
                    "height": height
                }
                yolo_annotations.append(annotation)

    if detected_boxes == 0:
        logger.warning("Keine Objekte in YOLO-Detektion gefunden.")
        return None, None, None, None

    return raw_image_path, yolo_annotations, width, height

def process_single_image(process_id, yolo_model, detection_labels, class_mapping, images_dir):
    """
    Verarbeitet ein einzelnes Bild und speichert die Dateien im angegebenen Verzeichnis.
    """
    image = scrape_image(process_id)
    if image is None:
        logger.warning(f"Bild für process_id {process_id} konnte nicht abgerufen werden.")
        return

    raw_image_path, image_annotations, width, height = process_image(
        image, yolo_model, detection_labels, process_id, class_mapping, images_dir
    )
    if raw_image_path and image_annotations:
        # Speichere die Annotationsdatei im selben Verzeichnis wie das Bild
        label_filename = f"{process_id}.txt"
        label_data = create_label_txt(image_annotations)
        label_path = save_label_txt(label_data, images_dir, label_filename)

        if label_path is None:
            logger.error(f"Fehler beim Speichern der Annotationsdatei für process_id: {process_id}")
            return

        logger.info(f"Bild und Annotation für process_id {process_id} gespeichert.")
    else:
        logger.warning(f"Keine Daten zum Speichern für process_id: {process_id}")

def main():
    # Definiere den Verzeichnispfad
    dataset_dir = "/private/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/ModelTraining/data"

    # Setze images_dir auf dataset_dir, da Bilder und Labels zusammen gespeichert werden sollen
    images_dir = dataset_dir

    # Erstelle das Verzeichnis, falls es nicht existiert
    os.makedirs(images_dir, exist_ok=True)

    yolo_model = YOLO('yolov8x-seg.pt')

    # Definiere die zu detektierenden Labels und die Klassenzuordnung
    detection_labels = ['boat']  # Ersetzen Sie durch Ihre Labels
    class_mapping = {label: idx for idx, label in enumerate(detection_labels)}

    # Liste von Bild-IDs oder Prozess-IDs
    process_ids = [
        1000, 1002, 1003, 1004, 1005, 1007, 1008, 1009,
        1010, 1011, 1012, 1013, 1014, 1015, 1016, 1019,
        1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028,
        1030,  1032, 1034, 1035, 1036, 1038, 1039,
        1040, 1044, 1047,
        1050, 1051, 1052, 1054, 1055, 1056, 1057, 1059,
        1061, 1062, 1063, 1064, 1066, 1067,
        1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079,
        1080,  1083, 1085, 1086, 1087,
        1091, 1092,  1094, 1096, 1097, 1098,
        1100, 1104, 1105, 1106, 1107, 1108, 1109,
        1110, 1111, 1113, 1114, 1115, 1116, 1117, 1118, 1119,
        1120, 1121, 1126,
        1130, 1131, 1133, 1134, 1135, 1136, 1138, 1139,
        1140, 1142, 1143, 1146, 1147, 1148,
        1150, 1151, 1153, 1154, 1155, 1156, 1159,
        1161, 1162, 1163, 1164, 1167, 1168, 1169,
        1170, 1171, 1174, 1175, 1177,
    ]

    for process_id in process_ids:
        process_single_image(
            process_id, yolo_model, detection_labels, class_mapping, images_dir
        )

if __name__ == "__main__":
    main()
