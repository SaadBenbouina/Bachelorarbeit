import os
import logging
import random

import requests
from PIL import Image
from io import BytesIO


# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Liste der User-Agent-Strings
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def construct_image_url(process_id):
    process_id_str = str(process_id).zfill(5)
    last_three_digits = process_id_str[-3:]
    reversed_digits = last_three_digits[::-1]
    path_digits = '/'.join(reversed_digits)
    image_url = f"https://www.shipspotting.com/photos/big/{path_digits}/{process_id}.jpg?cb=0"
    return image_url

def scrape_image(process_id):
    image_url = construct_image_url(process_id)
    logger.info(f"Konstruiertes Bild-URL: {image_url}")
    headers = {
        'User-Agent': get_random_user_agent(),
        'Referer': 'https://www.shipspotting.com',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    try:
        response = requests.get(image_url, headers=headers, timeout=10)
        if response.status_code == 200:
            image_data = response.content
            image = Image.open(BytesIO(image_data))
            return image
        else:
            logger.warning(f"Fehler beim Herunterladen des Bildes {image_url}: Statuscode {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Fehler beim Herunterladen des Bildes {image_url}: {e}")
        return None

def save_image(image_pil, path, filename):
    """
    Speichert das Bild im angegebenen Pfad.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = os.path.join(path, filename)
    try:
        image_pil.save(image_path)
        logger.info(f"Bild gespeichert unter: {image_path}")
        return image_path
    except Exception as e:
        logger.error(f"Fehler beim Speichern des Bildes unter: {image_path} - {e}")
        return None

def process_single_image(process_id, images_dir):
    """
    Verarbeitet ein einzelnes Bild und speichert es im angegebenen Verzeichnis.
    """
    image = scrape_image(process_id)
    if image is None:
        logger.warning(f"Bild für process_id {process_id} konnte nicht abgerufen werden.")
        return

    # Speichere das Bild
    raw_image_filename = f"{process_id}.jpg"
    raw_image_path = save_image(image, images_dir, raw_image_filename)
    if raw_image_path is None:
        logger.error(f"Fehler beim Speichern des Bildes für process_id: {process_id}")

def main():
    # Definiere den Verzeichnispfad
    dataset_dir = "/private/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/ModelTraining/dat55"

    # Setze images_dir auf dataset_dir, da Bilder und Labels zusammen gespeichert werden sollen
    images_dir = dataset_dir

    # Erstelle das Verzeichnis, falls es nicht existiert
    os.makedirs(images_dir, exist_ok=True)

    # Liste von Bild-IDs oder Prozess-IDs
    process_ids = [10234, 10987, 11234, 11567, 12234, 12567, 12987, 13234]

    for process_id in process_ids:
        process_single_image(
            process_id, images_dir
        )

if __name__ == "__main__":
    main()
