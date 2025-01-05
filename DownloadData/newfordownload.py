import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

from ultralytics import YOLO
from PIL import Image
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YOLO-Modell laden (bitte deinen Pfad anpassen!)
yolo_model = YOLO("/Users/saadbenboujina/Downloads/optuna_trial_2/weights/best.pt")

def scrape_and_download_images(url, download_dir="/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/ForCategory/dow", processed_dir="/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/ForCategory/train223/order"):
    """
    Öffnet eine Webseite via Selenium, sucht in der Tabelle mit id='tablepress-6'
    alle Bilder in Spalte 3 und lädt diese auf die Festplatte herunter.
    Im Anschluss werden die Bilder mit YOLO analysiert, Boote werden freigeschnitten.

    Parameter:
    ----------
    url : str
        Die URL der Webseite, auf der die Tabelle steht.
    download_dir : str
        Verzeichnis, in das die Originalbilder heruntergeladen werden.
    processed_dir : str
        Verzeichnis, in das die gecroppten Boat-Bilder gespeichert werden.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Headless-Modus
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=chrome_options)
    try:
        logger.info(f"Rufe Seite auf: {url}")
        driver.get(url)
        # Kurze Wartezeit, damit die Seite geladen ist
        time.sleep(2)

        # HTML-Quellcode von Selenium
        page_source = driver.page_source
    finally:
        driver.quit()

    # HTML mit BeautifulSoup parsen
    soup = BeautifulSoup(page_source, 'html.parser')

    # Tabelle mit id="tablepress-6" suchen
    table = soup.find('table', {'id': 'tablepress-6'})
    if not table:
        logger.warning("Tabelle mit id='tablepress-6' wurde nicht gefunden.")
        return

    tbody = table.find('tbody')
    if not tbody:
        logger.warning("Keine <tbody> in der Tabelle gefunden.")
        return

    rows = tbody.find_all('tr')
    if not rows:
        logger.warning("Keine Zeilen in der Tabelle gefunden.")
        return

    # Download-Ordner anlegen
    os.makedirs(download_dir, exist_ok=True)

    # Ordner für verarbeitete (gecroppte) Bilder anlegen
    os.makedirs(processed_dir, exist_ok=True)

    img_counter = 0

    # -------------------------------------------------------------------------
    # Durch jede Tabellen-Zeile iterieren
    for row_index, row in enumerate(rows, start=1):
        # Alle Zellen in der aktuellen Zeile
        columns = row.find_all('td')
        if len(columns) < 3:
            continue

        # Dritte Spalte (Spalten-Index 2)
        third_column = columns[2]

        # Alle img-Tags aus Spalte 3
        img_tags = third_column.find_all('img')
        if not img_tags:
            logger.info(f"Keine img-Tags in Zeile {row_index} gefunden.")
            continue

        # ---------------------------------------------------------------------
        # Für jede gefundene Bild-URL
        for img_tag in img_tags:
            img_url = img_tag.get('src')
            if not img_url:
                continue

            # Hier ggf. host/URL ergänzen, falls relative Pfade genutzt werden.
            # In deinem Beispiel sind es absolute URLs, daher wahrscheinlich nicht nötig.
            logger.info(f"Bild gefunden: {img_url}")

            try:
                response = requests.get(img_url, timeout=10)
                response.raise_for_status()

                # Speichere das Originalbild auf Platte
                img_counter += 1
                filename = f"image_{img_counter+40}.jpg"
                filepath = os.path.join(download_dir, filename)

                with open(filepath, 'wb') as f:
                    f.write(response.content)

                logger.info(f"Bild gespeichert unter: {filepath}")

                # -----------------------------------------------------------------
                # Nun YOLO-Inferenz für das heruntergeladene Bild
                process_image_with_yolo(filepath, processed_dir)

            except requests.RequestException as req_err:
                logger.error(f"Fehler beim Download von {img_url}: {req_err}")


def process_image_with_yolo(image_path, processed_boats_dir):
    """
    Führt eine YOLO-Erkennung durch, um 'Boats' im Bild zu identifizieren.
    Erkennt YOLO mehr als ein Boot (Confidence > 0.6), wird das Bild komplett
    übersprungen. Wenn genau ein Boot erkannt wird, wird das Boot aus dem Bild
    ausgeschnitten und in processed_boats_dir gespeichert.
    """
    try:
        # Bild laden
        original_image = Image.open(image_path).convert("RGB")

        # YOLO run
        results = yolo_model(image_path)
        detections_above_threshold = 0

        # Schauen, wie viele Boote > 0.6 confidence erkannt wurden
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = box.cls.item() if hasattr(box.cls, 'item') else box.cls
                confidence = box.conf.item() if hasattr(box.conf, 'item') else box.conf
                # Annahme: class 0 = 'Boat'
                if cls == 0 and confidence > 0.6:
                    detections_above_threshold += 1

        # Wenn mehr als ein Boot erkannt, Bild überspringen
        if detections_above_threshold > 1:
            logger.info(f"Überspringe {image_path}: {detections_above_threshold} Boote erkannt (> 0.6).")
            return

        # Wenn genau ein Boot erkannt, ausschneiden
        for result in results:
            boxes = result.boxes
            for idx, box in enumerate(boxes):
                cls = box.cls.item() if hasattr(box.cls, 'item') else box.cls
                confidence = box.conf.item() if hasattr(box.conf, 'item') else box.conf
                if cls == 0 and confidence > 0.6:
                    xyxy = box.xyxy.tolist()
                    if isinstance(xyxy[0], list):
                        xyxy = xyxy[0]
                    try:
                        x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                    except ValueError as conv_e:
                        logger.error(f"Fehler beim Konvertieren der Koordinaten: {conv_e}")
                        continue

                    # Randbedingungen prüfen
                    width, height = original_image.size
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    cropped_image = original_image.crop((x1, y1, x2, y2))

                    # Dateinamen vorbereiten
                    base_filename = os.path.splitext(os.path.basename(image_path))[0]
                    boat_filename = f"{base_filename}_boat_{idx}.jpg"
                    boat_path = os.path.join(processed_boats_dir, boat_filename)

                    try:
                        cropped_image.save(boat_path)
                        logger.info(f"Boot ausgeschnitten und gespeichert: {boat_path}")
                    except IOError as io_e:
                        logger.error(f"Fehler beim Speichern von {boat_path}: {io_e}")

    except Exception as e:
        logger.error(f"Fehler bei YOLO-Verarbeitung von {image_path}: {e}")


if __name__ == "__main__":
    # Beispielhaft
    test_url = "https://ship-spotting.de/schiffe/schiffstypen/schiffstypen-behoerdenschiffe/datenbank-hafenaufsicht/"
    scrape_and_download_images(test_url)
