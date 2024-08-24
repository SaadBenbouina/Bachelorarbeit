import os
import shutil
import fiftyone.zoo as foz

# Define the label and number of images to download
label_name = "Boat"
num_images = 10

# Load the Open Images dataset with the specific label
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=[label_name],
    max_samples=num_images,
)

# Set the directory where images and labels will be saved
output_dir = "/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/Data/Train"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over the samples and check available fields
for sample in dataset:
    print("Available fields:", sample.field_names)
    if "ground_truth" in sample.field_names:
        # Copy the image
        image_path = sample.filepath
        image_output_path = os.path.join(output_dir, os.path.basename(image_path))
        shutil.copy(image_path, image_output_path)

        # Save the label information in a compact format
        labels_output_path = os.path.join(output_dir, f"{os.path.basename(image_path)}.txt")

        # Ensure the file is opened and written to correctly
        with open(labels_output_path, 'w') as f:
            detections = sample.ground_truth.detections
            for detection in detections:
                if detection.label == label_name:
                    # Bounding box format: [xmin, ymin, width, height]
                    bbox = detection.bounding_box
                    bbox_str = f"{bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f}"
                    label_str = f"{detection.label} {bbox_str}\n"
                    f.write(label_str)
    else:
        print(f"No ground_truth field found in sample {sample.id}")

print(f"Downloaded {num_images} images with labels to the '{output_dir}' directory.")
