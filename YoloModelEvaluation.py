import os
import shutil
from ultralytics import YOLO

# Funktion zur Berechnung des IoU-Werts
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

# Konvertiert die YOLO-Annotationen in xyxy-Format (linke obere Ecke, rechte untere Ecke)
def yolo_to_xyxy(img_width, img_height, box):
    x_center, y_center, width, height = box
    x1 = (x_center - width / 2) * img_width
    y1 = (y_center - height / 2) * img_height
    x2 = (x_center + width / 2) * img_width
    y2 = (y_center + height / 2) * img_height
    return [x1, y1, x2, y2]

# Lade das trainierte Modell
model = YOLO("boat_detection_yolo_model_new2/weights/best.pt")  # ersetze durch den tatsächlichen Pfad zu deinem Modell

# Definiere den Pfad zu deinem Datensatz und den Zielordner für Bilder mit niedrigem IoU
data_path = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/TrainDataYolo/train"
low_iou_folder = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/TrainDataYolo/0.7"

# Erstelle den Ordner, falls er noch nicht existiert
os.makedirs(low_iou_folder, exist_ok=True)

# Führt Vorhersagen auf dem Datensatz durch, mit stream=True, um Speicherprobleme zu vermeiden
results = model.predict(source=data_path, save=False, stream=True)

# Iteriere durch die Ergebnisse und überprüfe den IoU-Wert
for result in results:
    img_height, img_width = result.orig_shape
    image_name = os.path.basename(result.path)

    # Annotationsdatei laden
    annotation_file = os.path.join(data_path, os.path.splitext(image_name)[0] + '.txt')
    if not os.path.exists(annotation_file):
        continue

    with open(annotation_file, 'r') as file:
        ground_truth_boxes = []
        for line in file.readlines():
            class_id, x_center, y_center, width, height = map(float, line.split())
            if int(class_id) == 0:  # Nur Boote berücksichtigen
                gt_box = yolo_to_xyxy(img_width, img_height, [x_center, y_center, width, height])
                ground_truth_boxes.append(gt_box)

    boats = [box.xyxy[0].tolist() for box in result.boxes if int(box.cls) == 0]

    # Proceed as before
    if boats and ground_truth_boxes:
        for boat_box in boats:
            iou_values = [calculate_iou(boat_box, gt_box) for gt_box in ground_truth_boxes]
            max_iou = max(iou_values) if iou_values else 0

            # Check if the maximum IoU value is below 0.8
            if max_iou < 0.7:
                # Move the corresponding image and its label file to the target folder
                shutil.move(result.path, os.path.join(low_iou_folder, image_name))
                shutil.move(annotation_file, os.path.join(low_iou_folder, os.path.basename(annotation_file)))
                print(f"Bild {image_name} und sein Label wurden verschoben, da der IoU unter 0.7 liegt.")
                break  # Move the image and label if any box does not match welly
print("Evaluierung abgeschlossen.")
