from ultralytics import YOLO

# Laden Sie das trainierte YOLOv8-Segmentierungsmodell
model = YOLO("/private/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/boat_segmentation/weights/best.pt")
# Pfad zu Ihren Testbildern

metrics = model.val(
    data='/private/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/ModelTraining/config.yaml',  # Pfad zu Ihrer Datenkonfigurationsdatei
    split='test',        # Verwenden Sie den 'test'-Split Ihres Datensatzes
    imgsz=640,           # Bildgröße
    conf=0.4,            # Konfidenzschwelle
    iou=0.5,             # IoU-Schwelle für NMS
    task='segment'       # Wichtig für Segmentierungsaufgaben
)

# Drucken Sie die Metriken aus
print(metrics)
