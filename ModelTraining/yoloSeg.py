from ultralytics import YOLO

# Laden Sie das YOLOv8-Segmentierungsmodell
model = YOLO('yolov8l-seg.pt')

# Training starten
model.train(
    data='config.yaml',
    epochs=30,         # Anzahl der Epochen
    imgsz=640,         # Bildgröße
    batch=8,          # Batch-Größe
    name='boat_segmentation',
    project="/Users/saadbenboujina/Desktop/best",
    task='segment',
    patience=8                   # Early Stopping: Stoppt, wenn keine Verbesserung nach 5 Epochen
)


""""
import optuna
from ultralytics import YOLO

# Ziel-Funktion für Optuna
def objective(trial):
    # Hyperparameter definieren und Grenzen angeben
    epochs = trial.suggest_int("epochs", 30, 100)          # Anzahl der Epochen
    batch_size = trial.suggest_int("batch", 4, 16)         # Batch-Größe
    learning_rate = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)  # Lernrate
    img_size = trial.suggest_categorical("imgsz", [320, 480, 640])    # Bildgröße

    # YOLO-Modell laden
    model = YOLO('yolov8l-seg.pt')

    # Training starten
    results = model.train(
        data='/content/config.yaml',         # Pfad zur YAML-Datei
        epochs=epochs,                       # Anzahl der Epochen
        batch=batch_size,                    # Batch-Größe
        imgsz=img_size,                      # Bildgröße
        lr0=learning_rate,                   # Lernrate
        name=f"optuna_trial_{trial.number}", # Name des Trials
        project="/content/optuna_results",   # Ergebnisse speichern
        task='segment',                      # Segmentierungsaufgabe
        patience=8                           # Early Stopping: Abbruch nach 8 Epochen ohne Verbesserung
    )

    # Validierungs-mAP (z. B. mAP@50) extrahieren
    val_map = results.metrics["val/map50"]  # Verwende mAP@50 als Ziel
    return val_map

# Optuna-Studie starten
study = optuna.create_study(direction="maximize")  # Ziel: mAP maximieren
study.optimize(objective, n_trials=15)            # 10 Versuche ausführen

# Beste Parameter ausgeben
print("Beste Parameter:", study.best_params)
print("Beste mAP:", study.best_value)

# Ergebnisse speichern
study.trials_dataframe().to_csv("/content/optuna_results.csv")
""""
