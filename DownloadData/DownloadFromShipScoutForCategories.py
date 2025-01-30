import os
import time
import requests
from bs4 import BeautifulSoup
from ultralytics import YOLO
from PIL import Image
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from DownloadData.ShipLabelFilter import ShipLabelFilter

# Log-Konfiguration: Gibt alle Meldungen ab INFO-Level aus
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard-Header für HTTP-Anfragen (Vermeidung von Sperren/Filtern)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, wie Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://www.shipspotting.com',
    'Accept-Language': 'en-US,en;q=0.9',
}

# YOLO-Modell laden
yolo_model = YOLO("/Users/saadbenboujina/Downloads/optuna_trial_2/weights/best.pt")

def scrape_page_selenium(process_id):
    """
    Ruft mittels Selenium die Seite von ShipSpotting.com auf und gibt den HTML-Quelltext zurück.
    """
    url = f"https://www.shipspotting.com/photos/{process_id}"
    logger.info(f"Öffne URL mit Selenium: {url}")

    # Selenium in Kopf-los (headless) Modus konfigurieren
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, wie Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(url)
        # Kurze Wartezeit, damit dynamische Inhalte geladen werden können
        time.sleep(0.5)
        return driver.page_source

    except Exception as e:
        logger.error(f"Fehler beim Abrufen von process_id {process_id} via Selenium: {e}")
        return None

    finally:
        driver.quit()

def extract_image_urls(soup):
    """
    Liest die Bild-URLs aus dem HTML-Code heraus und gibt sie als Liste zurück.
    """
    image_urls = []
    divs = soup.findAll('div', class_='summary-photo__image-row__image')
    for div in divs:
        img_tag = div.find('img')
        if img_tag and 'src' in img_tag.attrs:
            image_urls.append(img_tag['src'])
    return image_urls

def extract_labels(soup):
    """
    Liest die Kategorie-Labels aus dem HTML-Code heraus und nutzt ShipLabelFilter zur Bereinigung.
    """
    labels = []
    label_divs = soup.find_all(
        'div',
        class_='InformationItem__InfoItemStyle-sc-r4h3tv-0 hfSVPp information-item summary-photo__card-general__label'
    )
    for div in label_divs:
        title = div.find('span', class_='information-item__title')
        if title and title.text.strip() == "Photo Category:":
            label_value = div.find('span', class_='information-item__value')
            if label_value:
                label_text = label_value.text.strip()
                if label_text:
                    filtered_label = ShipLabelFilter.filter_label(label_text)
                    if filtered_label:
                        labels.append(filtered_label)
                    else:
                        logger.warning(f"Filter gab None zurück für label_text: '{label_text}'")
    return labels

def download_images_from_shipspotting(process_id, url_prefix='https://www.shipspotting.com/photos/'):
    """
    Ruft per Selenium die Seite auf, um Bild-URLs und Kategorie-Labels zu sammeln.
    """
    image_urls = []
    labels = []
    # Formatierte ID für ShipSpotting; ggf. anpassen, falls andere Form gewünscht
    photo_id = "{:07d}".format(process_id)
    url = url_prefix + photo_id

    logger.info(f"Lade Seite {url} via Selenium")

    page_source = scrape_page_selenium(process_id)
    if not page_source:
        logger.error(f"Kein Page-Quelltext für process_id {process_id}")
        return image_urls, labels

    soup = BeautifulSoup(page_source, 'html.parser')

    # URLs und Labels extrahieren
    image_urls = extract_image_urls(soup)
    labels = extract_labels(soup)

    return image_urls, labels

def save_images_into_categories(image_urls, labels, output_dir='downloaded_images', process_id=None):
    """
    Lädt die Bilder anhand der URLs herunter und speichert sie nach Kategorie in Unterordnern.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_paths = []
    for index, image_url in enumerate(image_urls):
        label = labels[index] if index < len(labels) else None
        if label is None:
            logger.info(f"Überspringe Bild index {index} (process_id {process_id}): Kein Label zugewiesen.")
            continue

        if not image_url:
            logger.warning(f"Keine valide Bild-URL für process_id {process_id} (index {index})")
            continue

        # Relative URL ergänzen, falls nötig
        if image_url.startswith('/'):
            image_url = 'https://www.shipspotting.com' + image_url

        logger.info(f"Lade Bild von {image_url}")
        try:
            response = requests.get(image_url, headers=HEADERS, timeout=10)
            response.raise_for_status()
        except requests.RequestException as req_e:
            logger.error(f"Fehler beim Herunterladen: {image_url} - {req_e}")
            continue

        # Define the category directory
        category_dir = os.path.join(output_dir, label[0])
        os.makedirs(category_dir, exist_ok=True)

        # Save the image in the category directory
        image_filename = f"image_{process_id}_{index}.jpg"
        image_path = os.path.join(category_dir, image_filename)
        try:
            with open(image_path, 'wb') as image_file:
                image_file.write(response.content)
            logger.info(f"Bild gespeichert unter {image_path}")
            saved_paths.append(image_path)
        except IOError as io_e:
            logger.error(f"Fehler beim Speichern: {image_path} - {io_e}")

    return saved_paths

def process_image_with_yolo(image_path, category, output_base_dir='processed_boats'):
    try:
        original_image = Image.open(image_path).convert("RGB")
        results = yolo_model(image_path)

        # Anzahl erkannter Boote ermitteln
        detections_above_threshold = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = box.cls.item() if hasattr(box.cls, 'item') else box.cls
                confidence = box.conf.item() if hasattr(box.conf, 'item') else box.conf
                if cls == 0 and confidence > 0.6:  # Klasse 0 = Boot
                    detections_above_threshold += 1

        #  Wenn mehr als 1 Boot erkannt wurde → Überspringen
        if detections_above_threshold > 1:
            logger.info(f"Überspringe Bild {image_path}: {detections_above_threshold} Boote gefunden.")
            return

        # Boote einzeln zuschneiden und abspeichern
        for result in results:
            boxes = result.boxes
            for idx, box in enumerate(boxes):
                cls = box.cls.item() if hasattr(box.cls, 'item') else box.cls
                confidence = box.conf.item() if hasattr(box.conf, 'item') else box.conf
                if cls == 0 and confidence > 0.6:
                    xyxy = box.xyxy.tolist()
                    # Falls xyxy[0] eine Liste ist, einmal auspacken
                    if isinstance(xyxy[0], list):
                        xyxy = xyxy[0]

                    # Versuche, die Koordinaten zu konvertieren
                    try:
                        x1, y1, x2, y2 = map(int, xyxy)
                    except Exception as conv_e:
                        logger.error(f"Fehler beim Koordinaten-Konvertieren: {conv_e}")
                        continue

                    # Grenzen einhalten
                    width, height = original_image.size
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)

                    cropped_image = original_image.crop((x1, y1, x2, y2))

                    # Kategorieverzeichnis erstellen
                    category_dir = os.path.join(output_base_dir, category if category else "Unknown")
                    os.makedirs(category_dir, exist_ok=True)

                    base_filename = os.path.splitext(os.path.basename(image_path))[0]
                    boat_filename = f"{base_filename}_boat_{idx}.jpg"
                    boat_path = os.path.join(category_dir, boat_filename)
                    try:
                        cropped_image.save(boat_path)
                        logger.info(f"Ausschnitt gespeichert: {boat_path}")
                    except IOError as io_e:
                        logger.error(f"Fehler beim Speichern des Ausschnitts {boat_path}: {io_e}")

    except Exception as e:
        logger.error(f"Fehler bei YOLO-Verarbeitung für {image_path}: {e}")

def main():
    """
    Hauptablauf:
    - Definiert Start- und End-ID
    - Lädt Bilder und Labels von ShipSpotting
    - Speichert sie in Kategoriefolder
    """
    start_id = 274385
    end_id = 275389
    download_folder = "RawPicturesForCategory"
    processed_boats_dir = "Kategory/train"

    for process_id in range(start_id, end_id):
        logger.info(f"Bearbeite ID: {process_id}")

        # Seite aufrufen, Bilder + Labels finden
        image_urls, labels = download_images_from_shipspotting(process_id)
        if not image_urls or not labels:
            logger.warning(f"Keine Bilder oder Labels für ID {process_id}. Überspringe.")
            continue

        # Bilder in Ordnern nach Kategorien speichern
        saved_image_paths = save_images_into_categories(image_urls, labels, download_folder, process_id)

        # Mit YOLO verarbeiten und Boot-Ausschnitte in Kategorien speichern
        for idx, image_path in enumerate(saved_image_paths):
            category = labels[idx] if idx < len(labels) else "Unknown"
            process_image_with_yolo(image_path, category, output_base_dir=processed_boats_dir)

if __name__ == '__main__':
    main()
