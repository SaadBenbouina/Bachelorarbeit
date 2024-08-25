import os
import shutil

# Quellverzeichnis, das die .txt-Dateien enthält
source_dir = "/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/Data/Train/images"

# Zielverzeichnis, in das die .txt-Dateien verschoben werden sollen
destination_dir = "/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/Data/Train/labels"

# Stelle sicher, dass das Zielverzeichnis existiert, ansonsten erstelle es
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Durchlaufe alle Dateien im Quellverzeichnis
for filename in os.listdir(source_dir):
    if filename.endswith(".txt"):
        # Vollständiger Pfad zur Quelldatei
        source_file = os.path.join(source_dir, filename)

        # Vollständiger Pfad zur Zieldatei
        destination_file = os.path.join(destination_dir, filename)

        # Verschiebe die Datei
        shutil.move(source_file, destination_file)
        print(f"Verschoben: {filename}")

print("Alle .txt-Dateien wurden erfolgreich verschoben.")
