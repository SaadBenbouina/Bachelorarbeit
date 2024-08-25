import os

# Verzeichnis, das die .jpg-Dateien enthält
input_dir = "/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/Data/Train"

# Ersetze 'Boat' durch die numerische Klassen-ID (z.B. 0)
class_mapping = {
    "Boat": 0
}

def convert_to_yolo_format(input_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            # Ersetze die Dateiendung .jpg durch .txt
            txt_filename = filename.replace(".jpg", ".txt")
            txt_file_path = os.path.join(input_dir, txt_filename)

            if not os.path.exists(txt_file_path):
                print(f"Warnung: {txt_filename} existiert nicht.")
                continue

            with open(txt_file_path, 'r') as file:
                lines = file.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()

                # Ersetze den Klassennamen durch die numerische ID
                class_name = parts[0]
                class_id = class_mapping.get(class_name, -1)  # Gibt -1 zurück, falls der Klassenname nicht gefunden wird

                if class_id == -1:
                    print(f"Warnung: Klasse {class_name} in Datei {txt_filename} nicht gefunden.")
                    continue

                # Normierte Werte bleiben gleich, außer der Klassennamen wird durch ID ersetzt
                new_line = f"{class_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n"
                new_lines.append(new_line)

            # Schreibe die neue Datei im YOLO-Format
            with open(txt_file_path, 'w') as file:
                file.writelines(new_lines)

            print(f"{txt_filename} wurde erfolgreich konvertiert.")

# Führt die Konvertierung durch
convert_to_yolo_format(input_dir)
