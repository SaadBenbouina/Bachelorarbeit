import os
import logging
import random
import requests
from PIL import Image
from io import BytesIO

# Log-Konfiguration: Stellt sicher, dass Meldungen ab INFO-Level ausgegeben werden
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Liste mit User-Agent-Strings zur Umgehung möglicher Sperren
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, wie Gecko) '
    'Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, wie Gecko) '
    'Version/14.0.3 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, wie Gecko) '
    'Chrome/91.0.4472.114 Safari/537.36',
]

def get_random_user_agent():
    """
    Gibt zufällig einen Eintrag aus USER_AGENTS zurück,
    um bei jeder Anfrage einen unterschiedlichen User-Agent zu verwenden.
    """
    return random.choice(USER_AGENTS)

def construct_image_url(process_id):
    """
    Baut aus einer Bild-ID (process_id) den spezifischen URL zusammen,
    unter dem das Bild abgerufen werden kann.
    """
    # process_id als string mit 5 Stellen (z.B. '00123')
    pid_str = str(process_id).zfill(5)
    # Letzte 3 Ziffern extrahieren
    last_three = pid_str[-3:]
    # Diese 3 Ziffern umkehren
    reversed_digits = last_three[::-1]
    # Für den URL müssen die Ziffern mit '/' getrennt werden
    path_digits = '/'.join(reversed_digits)
    # URL zusammensetzen
    return f"https://www.shipspotting.com/photos/big/{path_digits}/{process_id}.jpg?cb=0"

def scrape_image(process_id):
    """
    Lädt das Bild anhand der construct_image_url-Funktion herunter,
    gibt ein PIL-Image-Objekt zurück oder None bei Fehler.
    """
    url = construct_image_url(process_id)
    logger.info(f"Rufe Bild ab: {url}")
    headers = {
        'User-Agent': get_random_user_agent(),
        'Referer': 'https://www.shipspotting.com',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    try:
        # Sendet GET-Request
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            # Erstellt aus dem heruntergeladenen Byte-Strom ein PIL-Image
            return Image.open(BytesIO(response.content))
        else:
            logger.warning(f"Statuscode {response.status_code} für {url}")
            return None
    except Exception as e:
        logger.error(f"Fehler beim Herunterladen von {url}: {e}")
        return None

def save_image(image_pil, path, filename):
    """
    Speichert ein PIL-Image unter 'path/filename'.
    Erstellt das Verzeichnis, falls es noch nicht existiert.
    Gibt den vollständigen Pfad zurück oder None, falls ein Fehler auftritt.
    """
    # Ordner anlegen, falls er nicht existiert
    if not os.path.exists(path):
        os.makedirs(path)
    # Vollständiger Pfad inkl. Dateiname
    full_path = os.path.join(path, filename)
    try:
        # Bild im Dateisystem speichern
        image_pil.save(full_path)
        logger.info(f"Bild gespeichert: {full_path}")
        return full_path
    except Exception as e:
        logger.error(f"Fehler beim Speichern: {full_path} - {e}")
        return None

def process_single_image(process_id, images_dir):
    """
    Versucht, ein Bild anhand einer ID herunterzuladen und
    speichert es anschließend im angegebenen Verzeichnis.
    """
    image = scrape_image(process_id)
    if image is None:
        logger.warning(f"Kein Bild für ID {process_id}.")
        return
    # Dateiname aus process_id zusammensetzen und speichern
    save_image(image, images_dir, f"{process_id}.jpg")

def main():
    """
    Hauptfunktion:
    - Definiert das Zielverzeichnis
    - Erstellt eine Liste mit IDs, für die Bilder heruntergeladen werden sollen
    - Ruft process_single_image für jede ID auf
    """
    # Pfad für das Zielverzeichnis
    dataset_dir = "Pictures"
    os.makedirs(dataset_dir, exist_ok=True)

    # IDs der zu verarbeitenden Bilder
    process_ids = [
        1624, 168, 169, 16297, 16304, 16308, 16309, 16311, 16314, 16322, 16325, 16326, 16332, 16157,
        16161, 16168, 16192, 16224, 16225, 16239, 16251, 16260, 16265, 16268, 16273, 16279, 15999, 16011,
    ]

    # Schleife über alle IDs, um die Bilder herunterzuladen und zu speichern
    for pid in process_ids:
        process_single_image(pid, dataset_dir)

# Sorgt dafür, dass main() nur ausgeführt wird, wenn dieses Skript direkt aufgerufen wird
if __name__ == "__main__":
    main()
