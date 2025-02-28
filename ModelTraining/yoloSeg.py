import optuna
from ultralytics import YOLO


def objective(trial):
    # Hyperparameter definieren und Grenzen angeben
    epochs = trial.suggest_int("epochs", 30, 60)  # Anzahl der Epochen
    batch_size = trial.suggest_int("batch", 4, 16, 32)  # Batch-Größe
    learning_rate = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)  # Lernrate
    img_size = trial.suggest_categorical("imgsz", [480, 640, 800])  # Bildgröße

    # YOLO-Modell laden
    model = YOLO('yolov8l-seg.pt')

    try:
        # Training starten
        results = model.train(
            data='/content/drive/MyDrive/Colab Notebooks/config.yaml',  # Pfad zur YAML-Datei
            epochs=epochs,  # Anzahl der Epochen
            batch=batch_size,  # Batch-Größe
            imgsz=img_size,  # Bildgröße
            lr0=learning_rate,  # Lernrate
            name=f"optuna_trial_{trial.number}",  # Name des Trials
            project="/content/drive/MyDrive/optuna_resultsnew55",  # Ergebnisse speichern
            task='segment',  # Segmentierungsaufgabe
            patience=8,  # Early Stopping: Abbruch nach 8 Epochen ohne Verbesserung
            optimizer='auto',
        )

        # Validierungs-mAP (z. B. mAP@50) extrahieren
        val_map = results["metrics/seg_mAP50"]
        return val_map

    except Exception as e:
        # Fehlerbehandlung, um sicherzustellen, dass Optuna den Fehler erkennt
        print(f"Trial {trial.number} failed due to {e}")
        return float("nan")  # NaN zurückgeben, wenn ein Fehler auftritt


# Optuna-Studie starten
study = optuna.create_study(direction="maximize")  # Ziel: mAP maximieren
study.optimize(objective, n_trials=15)  # 15 Versuche ausführen

# Beste Parameter ausgeben
print("Beste Parameter:", study.best_params)
print("Beste mAP:", study.best_value)

# Ergebnisse speichern
study.trials_dataframe().to_csv("/content/drive/MyDrive/optuna_results55.csv")
