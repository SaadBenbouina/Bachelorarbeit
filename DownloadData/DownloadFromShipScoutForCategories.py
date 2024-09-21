import os
import requests
from bs4 import BeautifulSoup
from ultralytics import YOLO
from PIL import Image
import io
import xml.etree.ElementTree as ET
import logging

# Konfigurieren Sie das Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definieren Sie die Header für die HTTP-Anfragen
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, wie Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://www.shipspotting.com',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Laden Sie das YOLO-Modell
yolo_model = YOLO("../YoloModel/boat_detection_yolo_model_new3/weights/best.pt")

def download_images_from_shipspotting(process_id, url_prefix='https://www.shipspotting.com/photos/'):
    image_urls = []
    labels = []
    photo_id = "{:07d}".format(process_id)
    url = url_prefix + photo_id

    logger.info(f"Fetching URL: {url}")

    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            logger.error(f"Error: Received status code {response.status_code} from {url}")
            return image_urls, labels

        soup = BeautifulSoup(response.text, 'html.parser')

        # Finden Sie alle Bildcontainer auf der Seite
        divs = soup.findAll('div', class_='summary-photo__image-row__image')
        for div in divs:
            img_tag = div.find('img')
            if img_tag and 'src' in img_tag.attrs:
                image_urls.append(img_tag['src'])

        # Finden Sie die "Photo Category" Labels
        label_divs = soup.find_all('div',
                                   class_='InformationItem__InfoItemStyle-sc-r4h3tv-0 hfSVPp information-item summary-photo__card-general__label')
        for div in label_divs:
            information_title = div.find('span', class_='information-item__title')
            if information_title and information_title.text.strip() == "Photo Category:":
                label_value = div.find('span', class_='information-item__value')
                if label_value:
                    labels.append(label_value.text.strip())

    except Exception as e:
        logger.error(f"Error retrieving images for process_id {process_id}: {e}")

    return image_urls, labels

def save_images_and_labels(image_urls, labels, output_dir='downloaded_images', process_id=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_paths = []
    try:
        for index, image_url in enumerate(image_urls):
            if image_url:
                logger.info(f"Downloading image from {image_url}")
                image_data = requests.get(image_url, headers=HEADERS).content

                image_filename = f'image_{process_id}_{index}.jpg'
                image_path = os.path.join(output_dir, image_filename)
                with open(image_path, 'wb') as image_file:
                    image_file.write(image_data)
                logger.info(f"Image saved to {image_path}")
                saved_paths.append(image_path)
            else:
                logger.warning(f"No valid image URL provided for process_id {process_id}")
    except Exception as e:
        logger.error(f"Error saving image or label for {process_id}: {e}")

    return saved_paths

def generate_xml(image_path, category, output_dir='processed_boats'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.basename(image_path)
    xml_filename = os.path.splitext(filename)[0] + '.xml'
    xml_path = os.path.join(output_dir, xml_filename)

    # Erstellen Sie die XML-Struktur
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'filename').text = filename
    ET.SubElement(annotation, 'category').text = category

    # Optional: Weitere Informationen hinzufügen, z.B. Bounding Box
    # Dies erfordert jedoch, dass Sie die Bounding Box-Koordinaten übergeben
    # und die generate_xml-Funktion entsprechend anpassen

    # Schreiben Sie die XML-Datei
    tree = ET.ElementTree(annotation)
    tree.write(xml_path)
    logger.info(f"XML file saved to {xml_path}")

def process_image_with_yolo(image_path, category, output_dir='processed_boats'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Laden Sie das Bild
        image = Image.open(image_path).convert("RGB")

        # Führen Sie die YOLO-Detektion durch, indem Sie den Bildpfad übergeben
        results = yolo_model(image_path)  # Alternativ: yolo_model(image)

        for result in results:
            boxes = result.boxes  # Annahme: result.boxes enthält die Bounding Boxes
            for idx, box in enumerate(boxes):
                # Konvertieren Sie box.cls von Tensor zu Python-Skalare
                cls = box.cls.item() if hasattr(box.cls, 'item') else box.cls

                if cls == 0:  # Annahme: Klasse 0 ist 'Boat'
                    # Debugging: Ausgabe der Struktur von box.xyxy.tolist()
                    xyxy = box.xyxy.tolist()
                    logger.debug(f"Original box.xyxy.tolist(): {xyxy}")

                    # Überprüfen, ob xyxy eine Liste von Listen ist
                    if isinstance(xyxy[0], list):
                        xyxy = xyxy[0]
                        logger.debug(f"Flattened box.xyxy.tolist(): {xyxy}")

                    # Konvertieren Sie box.xyxy von Tensor zu Python-Liste und dann zu Integers
                    try:
                        x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                    except Exception as conv_e:
                        logger.error(f"Error converting coordinates to integers: {conv_e}")
                        continue  # Überspringen Sie diese Box

                    # Optional: Überprüfen Sie die Bildgrenzen
                    width, height = image.size
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    cropped_image = image.crop((x1, y1, x2, y2))

                    # Speichern Sie das ausgeschnittene Bild
                    base_filename = os.path.splitext(os.path.basename(image_path))[0]
                    boat_filename = f"{base_filename}_boat_{idx}.jpg"  # Verwenden Sie den Index als eindeutigen Identifikator
                    boat_path = os.path.join(output_dir, boat_filename)
                    cropped_image.save(boat_path)
                    logger.info(f"Cropped boat image saved to {boat_path}")

                    # Erstellen Sie die XML-Datei im selben Verzeichnis
                    generate_xml(boat_path, category, output_dir=output_dir)

    except Exception as e:
        logger.error(f"Error processing image {image_path} with YOLO: {e}")

def main():
    start_id = 1134410
    end_id = 1134414
    download_folder = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/ForCategory/RawData"
    processed_boats_dir = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/ForCategory/ProcessedBoats"

    for process_id in range(start_id, end_id):
        logger.info(f"Processing ID: {process_id}")

        # Schritt 1: Bilder und Labels herunterladen
        image_urls, labels = download_images_from_shipspotting(process_id)
        logger.info(f"Image URLs: {image_urls}")
        logger.info(f"Labels: {labels}")

        # Schritt 2: Bilder speichern
        saved_image_paths = save_images_and_labels(image_urls, labels, download_folder, process_id)
        logger.info(f"Saved Images: {saved_image_paths}")

        # Schritt 3: Bilder mit YOLO verarbeiten
        for idx, image_path in enumerate(saved_image_paths):
            if idx < len(labels):
                category = labels[idx]
            else:
                category = "Unknown"
            process_image_with_yolo(image_path, category, output_dir=processed_boats_dir)

if __name__ == '__main__':
    main()
