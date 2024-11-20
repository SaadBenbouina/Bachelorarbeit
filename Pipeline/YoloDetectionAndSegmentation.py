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
from Pipeline.ship_image_scraper import scrape_image_selenium as scrape_image  # Selenium-basierte Funktion importieren
import time

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

def process_image(image, yolo_model, detection_labels, model_classification, device, process_id):
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

        if confidence > 0.2 and label in detection_labels:
            detected_boxes += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_image = image.crop((x1, y1, x2, y2))
            label_pred = classify_and_log(cropped_image, model_classification, device)

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
        mask_element = ET.SubElement(image_metadata, "mask", label=label, source="YOLOv8")
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
    logger.info(f"Schiffkategorie ist : {label_pred}")
    return label_pred

def process_single_image(process_id, yolo_model, detection_labels, model_classification, device):
    image = scrape_image(process_id)
    if image is None:
        return None, None

    xml_data, image_path = process_image(image, yolo_model, detection_labels, model_classification, device, process_id)
    return xml_data, image_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Verwende Gerät: {device}")

    yolo_model = YOLO('yolov8x-seg.pt')
    # Überprüfen Sie, ob YOLO das Gerät setzen kann
    if hasattr(yolo_model, 'to'):
        yolo_model.to(device)
    else:
        logger.warning("YOLO-Modell unterstützt die `.to(device)` Methode nicht.")

    # Klassifikationsmodell laden
    num_classes = 16
    model_classification = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model_classification.fc.in_features
    model_classification.fc = nn.Linear(num_ftrs, num_classes)

    # Pfad zu den gespeicherten Gewichten
    model_save_path = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/ship_classification_resnet50.pth"
    if not os.path.exists(model_save_path):
        logger.error(f"Klassifikationsmodell nicht gefunden am Pfad: {model_save_path}")
        return

    state_dict = torch.load(model_save_path, map_location=device)
    model_classification.load_state_dict(state_dict)
    model_classification.to(device)
    model_classification.eval()

    detection_labels = ["boat"]
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
        """""
        1179,
        1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189,
        1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199,
        1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209,
        1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219,
        1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229,
        1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239,
        1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249,
        1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259,
        1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269,
        1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279,
        1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289,
        1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299,
        1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309,
        1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319,
        1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329,
        1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339,
        1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349,
        1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359,
        1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369,
        1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379,
        1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389,
        1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399
        """""
    ]
    for process_id in process_ids:
        logger.info(f"Verarbeite process_id: {process_id}")
        xml_data, image_path = process_single_image(process_id, yolo_model, detection_labels, model_classification, device)
        if xml_data:
            save_xml(xml_data, f"{process_id}_processed.xml", "output3")
            logger.info(f"Ergebnis gespeichert unter: output3/{process_id}_processed.xml")
        else:
            logger.warning(f"Keine Daten zum Speichern für process_id: {process_id}")

        time.sleep(2)  # 2 Sekunden Pause zwischen den Anfragen

if __name__ == "__main__":
    main()
