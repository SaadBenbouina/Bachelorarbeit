from ultralytics import YOLO

# Laden Sie das YOLOv8-Segmentierungsmodell
model = YOLO('/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/newData/weights/best.pt')  # Sie können auch 'yolov8s-seg.pt' oder ein anderes Modell verwenden

# Training starten
model.train(
    data='config.yaml',
    epochs=50,         # Anzahl der Epochen
    imgsz=640,         # Bildgröße
    batch=16,          # Batch-Größe
    name='boat_segmentation',
    project="/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit",
    task='segment'     # Wichtig für Segmentierungsaufgaben
)
