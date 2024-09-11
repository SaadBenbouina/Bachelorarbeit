from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset
import os

# Initialize a SegmentsDataset from the release file
client = SegmentsClient('206566e9d1ca70533b6c069785b3f54b135d05e9')
release = client.get_release('smsabenb/boat2-semantic', 'vvv')  # Specify your release version
dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])

# Specify the custom output folder
custom_output_dir = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit"


# Ensure the output folder exists
os.makedirs(custom_output_dir, exist_ok=True)

# Export to COCO panoptic format, specifying the output folder
export_dataset(dataset, export_format='coco-panoptic', export_dir=custom_output_dir)

print(f"Dataset exported to {custom_output_dir}")
