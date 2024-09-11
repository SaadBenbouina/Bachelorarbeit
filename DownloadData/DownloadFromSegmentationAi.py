from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset
import os

# Initialize a SegmentsDataset from the release file
client = SegmentsClient('206566e9d1ca70533b6c069785b3f54b135d05e9')
release = client.get_release('smsabenb/boat2', 'test2')  # Specify your release version
dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])



# Ensure the output folder exists

# Export to COCO panoptic format, specifying the output folder
export_dataset(dataset, export_format='coco-panoptic', export_dir=custom_output_dir)

