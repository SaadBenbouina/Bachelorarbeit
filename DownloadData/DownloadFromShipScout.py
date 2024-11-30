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
    process_ids = [7592,167913,178024,23579,390246,656802,712468,1101357,1545791,1545791,1556802,1612468,1623579,1678024,1701357,1923579,2012468,2112468,2145791,2967913,2978024,3001357,3034680,3189135,3256802,3267913,3301357,3467913,3512468,3523579,3745791,2941,11234,12234,29901,42234,43456,44567,58890,65567,87789,88890,9901,94456,96678,1002234,1004456,1016678,1046678,1047789,1050012,1052234,1056678,1060012,1061123,1077789,1105567,1109901,1121123,1146678,1147789,1151123,11662234,1177789,1186678,1184456,1191123,1209901,12114456,1261123,1276678,1281123,1313345,1314456,1326678,1327789,1333345,1338890,1340012,1342234,1342234,1348890,1361123,1369901,1374456,1384456,1388890,1402234,1415567,1421123,1432234,1436678,1453345,1469901,1474456,1483345,1484456,1485567,1498890,1500012,1524456,1576678,15833345,1598890,1610012,163234,1639901,1649901,1665567,1700012,1707789,1720012,1726678,1731123,1737789,1743345,1781123,1790012,1793312,1793345,1796678,1810012,1813345,1816678,1848890,1849901,1852234,1856678,1886678,1903345,1912234,1914456,1915567,1945567,1951123,1975567,1980012,1983345,193345]

    for process_id in process_ids:
        process_single_image(
            process_id, images_dir
        )

if __name__ == "__main__":
    main()
