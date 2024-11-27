import numpy as np
import xml.etree.ElementTree as ET
import pycocotools.mask as mask_util
import random
import cv2
import torchvision.transforms as transforms
import torch
from torchvision import models
import torch.nn as nn
import os
import logging
from Pipeline.Mapper import map_number_to_ship
from Pipeline.ship_image_scraper import scrape_image_selenium as scrape_image  # Selenium-basierte Funktion importieren
import time
import torchvision

# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_unique_color():
    return np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], dtype=np.uint8)

def mask_to_rle(mask):
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle

def save_image(image_np, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = os.path.join(path, filename)
    cv2.imwrite(image_path, image_np)
    return image_path

def process_image(image, mask_rcnn_model, detection_labels, model_classification, device, process_id):
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    # Bildvorbereitung für Mask R-CNN
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).to(device)

    # Mask R-CNN Vorhersage
    with torch.no_grad():
        predictions = mask_rcnn_model([image_tensor])

    predictions = predictions[0]
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']
    masks = predictions['masks']

    detected_boxes = 0
    boxes_data = []
    drawn_masks = []

    # COCO-Klassen
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
        'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    threshold = 0.8  # Erhöhte Konfidenzschwelle
    for i in range(len(scores)):
        if scores[i] >= threshold:
            class_id = labels[i].item()
            label = COCO_INSTANCE_CATEGORY_NAMES[class_id]

            # Überprüfen, ob das erkannte Objekt in den gewünschten Labels ist
            if label in detection_labels:
                detected_boxes += 1
                x1, y1, x2, y2 = boxes[i].cpu().numpy().astype(int)

                # Bildausschnitt für die Klassifikation
                cropped_image = image.crop((x1, y1, x2, y2))
                label_pred = classify_and_log(cropped_image, model_classification, device)

                confidence = scores[i].item()
                label_text = f'{label_pred} {confidence:.2f}'

                # Zeichnen der Bounding Box und des Labels
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_np, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                boxes_data.append({
                    'label': label_pred,
                    'confidence': confidence,
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2
                })

                # Maske verarbeiten
                mask = masks[i, 0].cpu().numpy()
                mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_resized = mask_resized > 0.5  # Binärmaske

                color = generate_unique_color()
                image_np[mask_resized] = (image_np[mask_resized] * 0.5 + color * 0.5).astype(np.uint8)

                drawn_masks.append((confidence, label_pred, mask_resized))

    if detected_boxes == 0:
        logger.warning("Keine Objekte in Mask R-CNN-Detektion gefunden.")
        return None, None

    # Speichere das verarbeitete Bild und die XML-Daten
    image_path = save_image(image_np, "output3", f"{process_id}_processed.jpg")
    image_metadata = ET.Element("image", id=str(process_id), width=str(width), height=str(height))

    for box in boxes_data:
        box_element = ET.SubElement(image_metadata, "Bounding_box", label=box['label'], confidence=f"{box['confidence']:.2f}")
        box_element.set("x1", str(box['x1']))
        box_element.set("y1", str(box['y1']))
        box_element.set("x2", str(box['x2']))
        box_element.set("y2", str(box['y2']))

    for confidence, label, mask in drawn_masks:
        rle = mask_to_rle(mask)
        mask_element = ET.SubElement(image_metadata, "mask", label=label, source="Mask R-CNN")
        rle_str = ', '.join(str(count) for count in rle["counts"])
        mask_element.set("rle", rle_str)
        mask_element.set("confidence", f"{confidence:.2f}")

    return image_metadata, image_path

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

def save_xml(xml_data, file_name="output3.xml", path=""):
    if path:
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, file_name)
    else:
        file_path = file_name

    tree = ET.ElementTree(xml_data)
    ET.indent(tree, space="  ", level=0)
    tree.write(file_path)
    return file_path

def classify_and_log(image, model_classification, device):
    label_pred = map_number_to_ship(classify_image(image, model_classification, device))
    logger.info(f"Schiffkategorie ist: {label_pred}")
    return label_pred

def process_single_image(process_id, mask_rcnn_model, detection_labels, model_classification, device):
    image = scrape_image(process_id)
    if image is None:
        return None, None

    xml_data, image_path = process_image(image, mask_rcnn_model, detection_labels, model_classification, device, process_id)
    return xml_data, image_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Verwende Gerät: {device}")

    # Mask R-CNN-Modell laden (ohne Anpassung)
    mask_rcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    mask_rcnn_model.to(device)
    mask_rcnn_model.eval()

    # Klassifikationsmodell laden
    num_classes_classification = 16
    model_classification = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model_classification.fc.in_features
    model_classification.fc = nn.Linear(num_ftrs, num_classes_classification)

    # Pfad zu den gespeicherten Gewichten
    model_save_path = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/ship_classification_resnet50.pth"
    if not os.path.exists(model_save_path):
        logger.error(f"Klassifikationsmodell nicht gefunden am Pfad: {model_save_path}")
        return

    state_dict = torch.load(model_save_path, map_location=device)
    model_classification.load_state_dict(state_dict)
    model_classification.to(device)
    model_classification.eval()

    detection_labels = ["boat"]  # Verwenden Sie die COCO-Klassenbezeichnung
    process_ids = [
        1000, 1002, 1003, 1004, 1005, 1007, 1008, 1009,
        1010, 1011, 1012, 1013, 1014, 1015, 1016, 1019,

    ]
    for process_id in process_ids:
        logger.info(f"Verarbeite process_id: {process_id}")
        xml_data, image_path = process_single_image(process_id, mask_rcnn_model, detection_labels, model_classification, device)
        if xml_data:
            save_xml(xml_data, f"{process_id}_processed.xml", "output3")
            logger.info(f"Ergebnis gespeichert unter: output3/{process_id}_processed.xml")
        else:
            logger.warning(f"Keine Daten zum Speichern für process_id: {process_id}")

        time.sleep(2)  # 2 Sekunden Pause zwischen den Anfragen

if __name__ == "__main__":
    main()
