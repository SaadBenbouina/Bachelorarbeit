from ultralytics import YOLO

# Laden Sie das YOLOv8-Segmentierungsmodell
model = YOLO('yolov8x-seg.pt')

# Training starten
model.train(
    data='config.yaml',
    epochs=20,         # Anzahl der Epochen
    imgsz=640,         # Bildgröße
    batch=8,          # Batch-Größe
    name='boat_segmentation',
    project="/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit",
    task='segment',
    optimizer='AdamW',           # Optimierungsalgorithmus AdamW
    patience=5                   # Early Stopping: Stoppt, wenn keine Verbesserung nach 5 Epochen
)
