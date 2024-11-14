import requests
from bs4 import BeautifulSoup
import logging
from PIL import Image
import io

logger = logging.getLogger(__name__)

def scrape_image(process_id):
    """
    Scrapt ein Bild von ShipSpotting.com basierend auf der process_id.
    """
    url_prefix = 'https://www.shipspotting.com/photos/'
    url = f"{url_prefix}{str(process_id).zfill(7)}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, wie Gecko)'
                      ' Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.shipspotting.com',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    session = requests.Session()
    try:
        response = session.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        divs = soup.find_all('div', class_='summary-photo__image-row__image')
        image_urls = [div.find('img')['src'] for div in divs if div.find('img')]

        if not image_urls:
            logger.warning(f"Kein Bild gefunden für process_id: {process_id}")
            return None

        image_url = image_urls[0]
        if image_url.startswith('/'):
            image_url = 'https://www.shipspotting.com' + image_url

        image_response = session.get(image_url, headers=headers)
        image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
        return image

    except Exception as e:
        logger.error(f"Fehler beim Abrufen des Bildes für process_id {process_id}: {e}")
        return None
    finally:
        session.close()
