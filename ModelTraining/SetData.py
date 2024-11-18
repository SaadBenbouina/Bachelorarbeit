import os

import numpy as np
import cv2
import torch
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

def process_image(image, yolo_model, detection_labels, device, process_id, class_mapping, images_dir, conf_threshold=0.5, iou_threshold=0.5):
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
    yolo_result = yolo_model.predict(image_np, task='segment', conf=conf_threshold, iou=iou_threshold)
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

def process_single_image(process_id, yolo_model, detection_labels, device, class_mapping, images_dir):
    """
    Verarbeitet ein einzelnes Bild und speichert die Dateien im angegebenen Verzeichnis.
    """
    image = scrape_image(process_id)
    if image is None:
        logger.warning(f"Bild für process_id {process_id} konnte nicht abgerufen werden.")
        return

    raw_image_path, image_annotations, width, height = process_image(
        image, yolo_model, detection_labels, device, process_id, class_mapping, images_dir
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Verwende Gerät: {device}")

    yolo_model = YOLO('yolov8x-seg.pt')

    # Definiere die zu detektierenden Labels und die Klassenzuordnung
    detection_labels = ['boat']  # Ersetzen Sie durch Ihre Labels
    class_mapping = {label: idx for idx, label in enumerate(detection_labels)}

    # Liste von Bild-IDs oder Prozess-IDs
    process_ids = [10234, 10987, 11234, 11567, 12234, 12567, 12987, 13234,
                   13567, 13987, 14234, 14567, 14987, 15234, 15567, 15987, 16234, 16567,
                   16987, 17234, 17567, 17987, 18234, 18567, 19234, 19987,
                   20234, 20567, 20987, 21234, 21567, 22567, 22987,
                   23567, 23987, 25234, 25567, 25987, 26234, 26567,
                   26987, 27234, 27567, 27987, 28234, 28567, 29234, 29567, 29987,
                   30234, 30567, 30987, 31234, 31567, 31987, 32234, 32567, 32987, 33234,
                   33567, 33987, 34234, 34567, 34987, 35234, 35567, 35987, 36234, 36567,
                   36987, 37234, 37567, 38234, 38567, 38987, 39234, 39567, 39987,
                   40234, 40567, 40987, 41234, 41567, 41987, 42234, 42567, 42987, 43234,
                   43567, 43987, 44567, 44987, 45234, 45567, 45987, 46234, 46567,
                   46987, 47234, 47567, 47987, 48567, 48987, 49234, 49567, 49987,
                   50234, 50987, 51234, 51567, 51987, 52234, 52567, 53234,
                   53567, 53987, 54234, 54987, 55234, 55567, 55987, 56234, 56567,
                   56987, 57234, 57987, 58234, 58567, 58987, 59234, 59567, 59987,
                   60234, 60567, 60987, 61234, 61567, 61987, 62234, 62567, 62987,
                   63567, 63987, 64234, 64567, 64987, 65234, 66234, 66567,
                   67234, 67987, 68234, 68567, 68987, 69234, 69567, 69987,
                   70234, 70567, 70987, 71234, 71567, 72234, 72567, 72987, 73234,
                   73987, 74234, 74567, 74987, 75234, 75567, 75987, 76234, 76567,
                   76987, 77234, 77567, 77987, 78234, 78567, 78987, 79234, 79567, 79987,
                   80234, 80987, 81234, 81567, 81987, 82234, 82987, 83234,
                   83567, 83987, 84567, 84987, 85234, 85567, 85987, 86234, 86567,
                   86987, 87234, 87987, 88234, 88567, 88987, 89234, 89567, 89987,
                   90234, 90567, 91234, 91567, 91987, 92234, 92567, 92987, 93234,
                   93567, 93987, 94234, 94567, 94987, 95234, 95567, 95987, 96234,
                   97234, 97567, 98234, 98567, 98987, 99234, 99567, 99987,
                   10023, 10345, 11023, 11345, 11678, 12023, 12678, 13023,
                   13345, 13678, 14023, 14345, 14678, 15023, 15345, 15678, 16023, 16345,
                   16678, 17023, 17345, 17678, 18023, 18345, 18678, 19023, 19345, 19678
        , 20345, 20678, 21023, 21345, 21678, 22023, 22345, 22678, 23023
        , 23678, 24023, 24345, 24678, 25023, 25345, 25678, 26023,
                   26678, 27023, 27345, 27678, 28023, 28345, 28678, 29023, 29345,
                   30023, 30345, 30678, 31023, 31345, 31678, 32678, 33023,
                   33345, 33678, 34023, 34345, 34678, 35023, 35345, 35678, 36023, 36345,
                   36678, 37023, 37345, 37678, 38023, 38345, 38678, 39345, 39678,
                   40023, 40345, 40678, 41023, 41345, 41678, 42345, 42678, 43023,
                   43345, 43678, 44023, 44345, 44678, 45023, 45345, 45678, 46023, 46345,
                   46678, 47345, 47678, 48023, 48678, 49023, 49345,
                   50023, 50678, 51023, 51345, 51678, 52345, 52678, 53023
                   ]  # Fügen Sie hier Ihre Prozess-IDs hinzu

    for process_id in process_ids:
        process_single_image(
            process_id, yolo_model, detection_labels, device, class_mapping, images_dir
        )

if __name__ == "__main__":
    main()