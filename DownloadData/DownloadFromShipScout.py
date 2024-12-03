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
    process_ids = [16284,16288,16289,16297,16304,16308,16309,16311,16314,16322,16325,16326,16332,16157,16161,16168,16192,16224,16225,16239,16251,16260,16265,16268,16273,16279,15999,16011,16027,16040,16049,16051,16080,16097,16112,16126,16136,16137,16141,16149,16154,15825,15826,15828,15836,15843,15852,15853,15860,15867,15897,15903,15903,15931,15974,15987,15589,15607,15644,15644,15645,15648,15657,15668,15668,15670,15675,15675,15684,15744,1784,15806,15811,15811,15820,15821,154401,15406,15419,15450,15483,15490,15498,15527,15529,15540,15540,15545,15554,15560,15560,15563,15230,15232,15238,15239,15240,15240,15260,15296,15305,15355,15365,15355,15365,15376,1586,15392,15399,15054,15055,15060,15060,15061,15063,15077,15090,15090,15095,15107,15115,15136,15146,15146,15156,15188,16341,16348,16361,16366,16412,16418,16421,16422,16428,16429,16436,16457,16465]
    for process_id in process_ids:
        process_single_image(
            process_id, images_dir
        )

if __name__ == "__main__":
    main()
