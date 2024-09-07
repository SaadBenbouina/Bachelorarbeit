import os
import requests
from bs4 import BeautifulSoup


# Define headers for the requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://www.shipspotting.com',
    'Accept-Language': 'en-US,en;q=0.9',
}


# Function to download images from shipspotting.com based on a process_id
def download_images_from_shipspotting(process_id, url_prefix='https://www.shipspotting.com/photos/'):
    image_urls = []
    labels = []
    # Format the process_id into a seven-digit number (e.g., 123 -> 0000123)
    photo_id = "{:07d}".format(process_id)
    # Construct the full URL by appending the photo_id to the base URL
    url = url_prefix + photo_id

    print(f"Fetching URL: {url}")

    try:
        # Send an HTTP GET request to the URL with the headers
        response = requests.get(url, headers=HEADERS)
        # Check if the request was successful (status code 200)
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} from {url}")
            return image_urls, labels  # Return if the request failed

        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all image containers on the HTML page
        divs = soup.findAll('div', class_='summary-photo__image-row__image')
        for div in divs:
            # Find the 'img' tag within the container
            img_tag = div.find('img')
            # Check if the 'src' attribute exists
            if img_tag and 'src' in img_tag.attrs:
                # Extract the image URL
                image_urls.append(img_tag['src'])

        # Find the "Photo Category" label
        label_divs = soup.find_all('div',
                                   class_='InformationItem__InfoItemStyle-sc-r4h3tv-0 hfSVPp information-item summary-photo__card-general__label')
        for div in label_divs:
            information_title = div.find('span', class_='information-item__title')
            if information_title and information_title.text.strip() == "Photo Category:":
                label_value = div.find('span', class_='information-item__value')
                if label_value:
                    # Extract the category text
                    label = label_value.text.strip()
                    labels.append(label)

    except Exception as e:
        # Catch and report errors that occur during execution
        print(f"Error retrieving images for process_id {process_id}: {e}")

    # Return the found image URLs and the labels
    return image_urls, labels


# Function to save the downloaded images and labels to disk
def save_images_and_labels(image_urls, labels, output_dir='downloaded_images', process_id=None):
    # Check if the output directory exists; if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_paths = []
    try:
        for index, image_url in enumerate(image_urls):
            # Check if a valid image URL is provided
            if image_url:
                print(f"Downloading image from {image_url}")
                # Download the image data with headers
                image_data = requests.get(image_url, headers=HEADERS).content

                # Create a unique filename using the process_id and index
                image_filename = f'image_{process_id}_{index}.jpg'
                # Combine the output directory with the filename
                image_path = os.path.join(output_dir, image_filename)
                # Open the file in write-binary mode and save the image data
                with open(image_path, 'wb') as image_file:
                    image_file.write(image_data)
                print(f"Image saved to {image_path}")
                saved_paths.append(image_path)

                # Save the corresponding label in a .txt file
                label_filename = f'image_{process_id}_{index}.txt'
                # label_path = os.path.join(output_dir, label_filename)
                # with open(label_path, 'w') as label_file:
                #    if labels and len(labels) > index:
                #        label_file.write(labels[index])
                #    else:
                #        label_file.write("No label found")
                # -print(f"Label saved to {label_path}")
            else:
                print(f"No valid image URL provided for process_id {process_id}")
    except Exception as e:
        # Catch and report errors that occur during the save operation
        print(f"Error saving image or label for {process_id}: {e}")

    return saved_paths


def main():
    # Iterate over a range of process_ids
    for i in range(1134410, 1136000):
        process_id = i
        output_folder = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/TrainDataYolo/FromShipScout"

        # Step 1: Download images from shipspotting.com
        image_urls, labels = download_images_from_shipspotting(process_id)
        print(f"Image URLs: {image_urls}")
        print(f"Labels: {labels}")

        # Step 2: Save images and labels locally with a unique name
        saved_paths = save_images_and_labels(image_urls, labels, output_folder, process_id)
        print(f"Saved Images and Labels: {saved_paths}")


if __name__ == '__main__':
    main()
