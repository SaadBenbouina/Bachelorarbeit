from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import logging
from PIL import Image
import io
import requests

logger = logging.getLogger(__name__)

def scrape_image_selenium(process_id):
    """
    Scrapt ein Bild von ShipSpotting.com basierend auf der process_id unter Verwendung von Selenium.
    """
    url = f"https://www.shipspotting.com/photos/{process_id}"
    logger.info(f"Öffne URL mit Selenium: {url}")

    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Headless-Modus
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, wie Gecko) "
                                "Chrome/91.0.4472.124 Safari/537.36")

    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(url)
        # Warten Sie, bis die Seite vollständig geladen ist (optional: explizite Wartezeiten hinzufügen)
        driver.implicitly_wait(10)

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Überprüfen Sie den korrekten Klassennamen für das Bild-Container-Div
        divs = soup.find_all('div', class_='summary-photo__image-row__image')
        if not divs:
            logger.warning(f"Keine Divs mit der Klasse 'summary-photo__image-row__image' gefunden für process_id: {process_id}")
            return None

        image_urls = [div.find('img')['src'] for div in divs if div.find('img')]

        if not image_urls:
            logger.warning(f"Kein Bild gefunden für process_id: {process_id}")
            return None

        image_url = image_urls[0]
        if image_url.startswith('/'):
            image_url = 'https://www.shipspotting.com' + image_url

        logger.info(f"Gefundene Bild-URL: {image_url}")

        # Herunterladen des Bildes mit requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, wie Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.shipspotting.com',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'image/webp,*/*;q=0.8',
        }
        image_response = requests.get(image_url, headers=headers, timeout=10)
        logger.info(f"Statuscode der Bild-Antwort: {image_response.status_code}")

        if image_response.status_code != 200:
            logger.warning(f"Bild konnte nicht abgerufen werden für process_id: {process_id} (Statuscode {image_response.status_code})")
            return None

        image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
        logger.info(f"Bild erfolgreich abgerufen für process_id: {process_id}")
        return image

    except Exception as e:
        logger.error(f"Fehler beim Scrapen des Bildes für process_id {process_id}: {e}")
        return None
    finally:
        driver.quit()
